#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <ur_rtde/rtde_control_interface.h>
#include <chrono>
#include <thread>
#include <memory>

using std::placeholders::_1;
using namespace std::chrono_literals;
using namespace ur_rtde;

class ServoJToFollowerNode : public rclcpp::Node
{
public:
    ServoJToFollowerNode() : Node("servo_j_to_follower_node")
    {
        this->declare_parameter<std::string>("robot_ip", "192.168.0.100");
        this->declare_parameter<double>("dt", 0.002);
        this->declare_parameter<double>("lookahead_time", 0.1);
        this->declare_parameter<double>("gain", 300.0);
        this->declare_parameter<double>("speed", 0.5);
        this->declare_parameter<double>("acceleration", 0.5);

        std::string robot_ip = this->get_parameter("robot_ip").as_string();
        dt_ = this->get_parameter("dt").as_double();
        lookahead_time_ = this->get_parameter("lookahead_time").as_double();
        gain_ = this->get_parameter("gain").as_double();
        speed_ = this->get_parameter("speed").as_double();
        acceleration_ = this->get_parameter("acceleration").as_double();

        try {
            rtde_control_ = std::make_unique<RTDEControlInterface>(robot_ip);
            RCLCPP_INFO(this->get_logger(), "Connected to UR robot at %s", robot_ip.c_str());
        } catch (const std::exception &e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to connect to UR robot: %s", e.what());
            throw;
        }

        traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            "/target/multi_joint_trajectory", 10,
            std::bind(&ServoJToFollowerNode::traj_callback, this, _1)
        );

        RCLCPP_INFO(this->get_logger(), "ServoJToFollowerNode started. dt=%.4f", dt_);
    }

private:
    std::unique_ptr<RTDEControlInterface> rtde_control_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_sub_;
    double dt_;
    double lookahead_time_;
    double gain_;
    double speed_;
    double acceleration_;

    void traj_callback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        if (msg->points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty trajectory.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received trajectory with %zu points.", msg->points.size());

        for (const auto& point : msg->points) {
            if (point.positions.size() != 6) {
                RCLCPP_ERROR(this->get_logger(), "Expected 6 joint positions, got %zu. Skipping point.",
                             point.positions.size());
                continue;
            }

            std::vector<double> q(point.positions.begin(), point.positions.end());

            rtde_control_->servoJ(q, speed_, acceleration_, dt_, lookahead_time_, gain_);

            std::this_thread::sleep_for(
                std::chrono::duration<double>(dt_)
            );
        }

        rtde_control_->servoStop();
        RCLCPP_INFO(this->get_logger(), "Trajectory execution complete.");
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ServoJToFollowerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
