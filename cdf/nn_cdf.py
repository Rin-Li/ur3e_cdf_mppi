# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import numpy as np
import os
import sys
import torch
import math
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CDF_DIR = Path(__file__).resolve().parent
_DATA_DIR = _CDF_DIR / "data"
_MODEL_DIR = _CDF_DIR / "model"

from cdf.mlp import MLPRegression
from ur3e.ur3e import URRobot
from rdf.ur_rdf import BPSDF

PI = math.pi
np.random.seed(10)

def training_monitor(func):
    """Decorator that logs GPU/CPU resource usage and best-loss tracking around training."""
    def wrapper(self, *args, **kwargs):
        import psutil
        from collections import deque

        print("="*60)
        print("Training Monitor Started")
        print("="*60)

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Initial GPU Memory: {start_memory:.2f} GB")

        cpu_percent_start = psutil.cpu_percent(interval=1)
        print(f"Initial CPU Usage: {cpu_percent_start:.1f}%")
        print("="*60)

        self.loss_history = deque(maxlen=100)
        self.best_loss = float('inf')
        self.best_epoch = 0

        try:
            result = func(self, *args, **kwargs)

            end_time = time.time()
            total_time = end_time - start_time

            print("\n" + "="*60)
            print("Training Completed Successfully")
            print("="*60)
            print(f"Total Training Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
            print(f"Best Loss: {self.best_loss:.6f} at Epoch {self.best_epoch}")

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                current_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"Peak GPU Memory: {peak_memory:.2f} GB")
                print(f"Final GPU Memory: {current_memory:.2f} GB")

            cpu_percent_end = psutil.cpu_percent(interval=1)
            print(f"Final CPU Usage: {cpu_percent_end:.1f}%")
            print("="*60)

            return result

        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("Training Interrupted by User")
            print("="*60)
            elapsed = time.time() - start_time
            print(f"Elapsed Time: {elapsed/60:.1f} minutes")
            if hasattr(self, 'best_loss'):
                print(f"Best Loss So Far: {self.best_loss:.6f} at Epoch {self.best_epoch}")
            print("="*60)
            raise

        except Exception as e:
            print("\n" + "="*60)
            print(f"Training Failed with Error: {str(e)}")
            print("="*60)
            raise

    return wrapper

class CDF:
    def __init__(self, device, load_data=True) -> None:
        self.device = device

        self.batch_x = 10
        self.batch_q = 100
        self.max_q_per_link = 300

        if load_data:
            self.raw_data = np.load(str(_DATA_DIR / 'data.npy'), allow_pickle=True).item()
            self.data = self.process_data(self.raw_data)
            self.len_data = len(self.data['k'])

        self.ur = URRobot(device)

    def process_data(self,data):
        import pytorch3d.ops 
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        for k in keys:
            if len(data[k]['q']) == 0:
                data.pop(k)
                continue
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            q_lib = torch.inf*torch.ones(self.max_q_per_link,6,6).to(self.device)
            for i in range(1,7):
                mask = (q_idx==i)
                if len(q[mask])>self.max_q_per_link:
                    fps_q = pytorch3d.ops.sample_farthest_points(q[mask].unsqueeze(0),K=self.max_q_per_link)[0]
                    q_lib[:,:,i-1] = fps_q.squeeze()
                elif len(q[mask])>0:
                    q_lib[:len(q[mask]),:,i-1] = q[mask]

            processed_data[k] = {
                'x':torch.from_numpy(data[k]['x']).float().to(self.device),
                'q':q_lib,
            }
        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'k':torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }

        torch.save(final_data, str(_DATA_DIR / 'data.pt'))
        return final_data
    
    def load_data(self,path):
        data = torch.load(path)
        return data

    def select_data(self):
        x = self.data['x']
        q = self.data['q']
        idx = torch.randint(0, len(x), (self.batch_x,))
        x_batch, q_lib = x[idx], q[idx]
        q_batch = self.sample_q()
        d, grad = self.decode_distance(q_batch, q_lib)
        return x_batch, q_batch, d, grad

    def decode_distance(self, q_batch, q_lib):
        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        d_tensor = torch.ones(batch_x,batch_q,6).to(self.device)*torch.inf
        grad_tensor  = torch.zeros(batch_x,batch_q,6,6).to(self.device)
        for i in range(6):
            q_lib_temp = q_lib[:,:,:i+1,i].reshape(batch_x*self.max_q_per_link,-1).unsqueeze(0).expand(batch_q,-1,-1)
            q_batch_temp = q_batch[:,:i+1].unsqueeze(1).expand(-1,batch_x*self.max_q_per_link,-1)
            d_norm = torch.norm((q_batch_temp - q_lib_temp),dim=-1).reshape(batch_q,batch_x,self.max_q_per_link)

            d_norm_min,d_norm_min_idx = d_norm.min(dim=-1)
            grad = torch.autograd.grad(d_norm_min.reshape(-1),q_batch_temp,torch.ones_like(d_norm_min.reshape(-1)),retain_graph=True)[0]
            grad_min_q = grad.reshape(batch_q, batch_x, self.max_q_per_link, -1).gather(2, d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, i+1))[:, :, 0, :]
            grad_tensor[:, :, :i+1, i] = grad_min_q.transpose(0, 1)
            d_tensor[:, :, i] = d_norm_min.transpose(0, 1)

        d, d_min_idx = d_tensor.min(dim=-1)
        grad_final = grad_tensor.gather(3,d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,6,6))[:,:,:,0]
        return d, grad_final
    
    def sample_q(self,batch_q = None):
        if batch_q is None:
            batch_q = self.batch_q
        q_sampled = self.ur.theta_min + torch.rand(batch_q,6).to(self.device)*(self.ur.theta_max-self.ur.theta_min)
        q_sampled.requires_grad = True
        return q_sampled
    
    def projection(self,q,d,grad):
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def train_nn(self, epoches=500):
        from collections import deque
        self.loss_history = deque(maxlen=100)
        self.best_loss = float('inf')
        self.best_epoch = 0

        # Input: concatenated [x, q] of shape (B, 9)
        model = MLPRegression(input_dims=9, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128], skips=[], act_fn=torch.nn.ReLU, nerf=True)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.01, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                x_batch,q_batch,d,gt_grad = self.select_data()

                x_inputs = x_batch.unsqueeze(1).expand(-1,self.batch_q,-1).reshape(-1,3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x,-1,-1).reshape(-1,6)

                inputs = torch.cat([x_inputs,q_inputs],dim=-1)
                outputs = d.reshape(-1,1)
                gt_grad = gt_grad.reshape(-1,6)
                weights = torch.ones_like(outputs).to(self.device)

                d_pred = model.forward(inputs)
                d_grad_pred = torch.autograd.grad(d_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,create_graph=True)[0]
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()
                dd_grad_pred = torch.autograd.grad(d_grad_pred, q_inputs, torch.ones_like(d_grad_pred), retain_graph=True, create_graph=True)[0]
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                d_loss = ((d_pred - outputs)**2 * weights).mean()

                # Combined loss: MSE + Eikonal + tension + gradient direction
                w0, w1, w2, w3 = 5.0, 0.01, 0.01, 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            current_loss = loss.item()
            self.loss_history.append(current_loss)
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = iter
                torch.save(model.state_dict(), str(_MODEL_DIR / 'model_best.pt'))

            if iter % 10 == 0:
                if len(self.loss_history) > 0:
                    recent_avg_loss = np.mean(list(self.loss_history))
                    recent_std_loss = np.std(list(self.loss_history))

                    gpu_mem_str = ""
                    if torch.cuda.is_available():
                        current_mem = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_str = f"\tGPU Mem: {current_mem:.2f}GB"

                    print(f"Epoch:{iter}\tMSE: {d_loss.item():.3f}\tEikonal: {eikonal_loss.item():.3f}\t"
                          f"Tension: {tension_loss.item():.3f}\tGradient: {gradient_loss.item():.3f}\t"
                          f"Total: {current_loss:.3f}\tAvg(100): {recent_avg_loss:.3f}±{recent_std_loss:.3f}\t"
                          f"Best: {self.best_loss:.3f}@{self.best_epoch}{gpu_mem_str}\t"
                          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        return model
    
    def inference(self, x, q, model):
        model.eval()
        x, q = x.to(self.device), q.to(self.device)
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 3)
        q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,6)
        inputs = torch.cat([x_cat,q_cat],dim=-1)
        cdf_pred = model.forward(inputs)
        return cdf_pred
    
    def inference_d_wrt_q(self, x, q, model, return_grad=True):
        cdf_pred = self.inference(x, q, model)
        d = cdf_pred.reshape(len(x), len(q)).min(dim=0)[0]
        if return_grad:
            grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True, create_graph=True)[0]
            return d, grad
        else:
            return d

    def eval_nn(self, model, num_iter=3):
        eval_time = False
        eval_acc = True
        if eval_time:
            x = torch.rand(100, 3).to(self.device)*torch.tensor([[0.8, 0.8, 0.4]]).to(self.device)-torch.tensor([[0.4, 0.4, 0.0]]).to(self.device)
            q = self.sample_q(batch_q=100)
            time_cost_list = []
            for i in range(100):
                t0 = time.time()
                d = self.inference_d_wrt_q(x, q, model, return_grad=False)
                t1 = time.time()
                grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True, create_graph=True)[0]
                q_proj = self.projection(q, d, grad)
                t2 = time.time()
                if i > 0:
                    time_cost_list.append([t1-t0, t2-t1])
            mean_time_cost = np.mean(time_cost_list, axis=0)
            print(f'inference time cost:{mean_time_cost[0]}\t projection time cost: {mean_time_cost[1]}')

        if eval_acc:
            bp_sdf = BPSDF(8, -1.0, 1.0, self.ur, device)
            bp_sdf_model = torch.load(str(_ROOT / 'rdf' / 'models' / 'BP_8.pt'), weights_only=False)

            res = []
            for i in range (1000):
                x = torch.rand(1,3).to(device)*torch.tensor([[0.8,0.8,0.4]]).to(device)-torch.tensor([[0.4,0.4,0.0]]).to(device)
                q = self.sample_q(batch_q=1000)
                for _ in range(num_iter):
                    d,grad = self.inference_d_wrt_q(x,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                q,grad = q.detach(),grad.detach()   # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
                
                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    def eval_nn_noise(self,model,num_iter = 3):
            bp_sdf = BPSDF(8,-1.0,1.0,self.ur,device)
            bp_sdf_model = torch.load(str(_ROOT / 'rdf' / 'models' / 'BP_8.pt'), weights_only=False)

            res = []
            for i in range (1000):
                x = torch.rand(1,3).to(device)*torch.tensor([[0.8,0.8,0.4]]).to(device)-torch.tensor([[0.4,0.4,0.0]]).to(device)
                noise = torch.normal(0,0.03,(1,3)).to(device)
                x_noise = x + noise
                q = self.sample_q(batch_q=1000)
                for _ in range (num_iter):
                    d,grad = self.inference_d_wrt_q(x_noise,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                q,grad = q.detach(),grad.detach()   # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
                
                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    def check_data(self):
        x_batch, q_batch, d, grad = self.select_data()
        q_proj = self.projection(q_batch, d, grad)

        import trimesh
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        for q0,q1 in zip(q_batch,q_proj[1]):
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.PointCloud(x_batch.data.cpu().numpy(),colors=[255,0,0]))
            robot_mesh0 = self.ur.get_forward_robot_mesh(pose, q0.unsqueeze(0))[0]
            robot_mesh0 = np.sum(robot_mesh0)
            robot_mesh0.visual.face_colors = [0,255,0,100]
            scene.add_geometry(robot_mesh0)
            robot_mesh1 = self.ur.get_forward_robot_mesh(pose, q1.unsqueeze(0))[0]
            robot_mesh1 = np.sum(robot_mesh1)
            robot_mesh1.visual.face_colors = [0,0,255,100]
            scene.add_geometry(robot_mesh1)
            scene.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdf = CDF(device)
    cdf.train_nn(epoches=30000)