#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <ur_rtde/rtde_receive_interface.h>
#include "custom_msgs/msg/float64_stamped.hpp"
using namespace ur_rtde;

class ReceiverFromFollowerNode : public rclcpp::Node
{
public:
    ReceiverFromFollowerNode() : Node("receiver_from_follower_node")
    {
        // Declare and get parameters
        this->declare_parameter<std::string>("robot_ip", "192.168.1.102");
        this->declare_parameter<double>("dt", 0.01);

        this->get_parameter("robot_ip", robot_ip_);
        this->get_parameter("dt", dt);

        // Initialize RTDE receive interface
        rtde_receive_ = std::make_unique<RTDEReceiveInterface>(robot_ip_);

        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/follower/joint_state", 10);
        // dont use
        pose_pub_  = this->create_publisher<geometry_msgs::msg::PoseStamped>("/follower/pose/base", 10);
        twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/follower/twist", 10);
        q6_pub_ = this->create_publisher<custom_msgs::msg::Float64Stamped>("/observation/state/q6", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(dt)),
            std::bind(&ReceiverFromFollowerNode::timer_callback, this)
        );
        system_clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);

        RCLCPP_INFO(this->get_logger(), "ReceiverFromFollowerNode started with IP: %s at %.4f s",
                    robot_ip_.c_str(), dt);
    }

private:
    std::string robot_ip_;
    double dt;
    std::unique_ptr<RTDEReceiveInterface> rtde_receive_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Clock::SharedPtr system_clock_;
    rclcpp::Publisher<custom_msgs::msg::Float64Stamped>::SharedPtr q6_pub_;
    void timer_callback()
    {
        // 1. 関節データ
        std::vector<double> joints = rtde_receive_->getActualQ();
        std::vector<double> velocities = rtde_receive_->getActualQd();

        if (joints.size() != 6) {
            RCLCPP_WARN(this->get_logger(), "Received invalid joint data.");
            return;
        }

        sensor_msgs::msg::JointState joint_msg;
        joint_msg.header.stamp = this->now();
        joint_msg.name = {
            "base", "shoulder", "elbow",
            "wrist1", "wrist2", "wrist3"
        };
        joint_msg.position = joints;
        joint_msg.velocity = velocities;
        joint_pub_->publish(joint_msg);

        auto q6_msg = custom_msgs::msg::Float64Stamped();
        q6_msg.data = joints[5]; 
        q6_msg.header.stamp = system_clock_->now();
        q6_msg.header.frame_id = "base_link";
        q6_pub_->publish(q6_msg);

        // 2. TCP Pose
        std::vector<double> tcp_pose = rtde_receive_->getActualTCPPose(); // [x, y, z, rx, ry, rz]
        if (tcp_pose.size() == 6) {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = this->now();
            pose_msg.header.frame_id = "base_link";

            pose_msg.pose.position.x = tcp_pose[0];
            pose_msg.pose.position.y = tcp_pose[1];
            pose_msg.pose.position.z = tcp_pose[2];

            // URのTCP姿勢(rx, ry, rz)（回転ベクトル）→クォータニオンに変換
            double angle = std::sqrt(tcp_pose[3]*tcp_pose[3] + tcp_pose[4]*tcp_pose[4] + tcp_pose[5]*tcp_pose[5]);
            geometry_msgs::msg::Quaternion quat;
            if (angle < 1e-6) {
                quat.x = quat.y = quat.z = 0.0;
                quat.w = 1.0;
            } else {
                double axis_x = tcp_pose[3] / angle;
                double axis_y = tcp_pose[4] / angle;
                double axis_z = tcp_pose[5] / angle;
                quat.x = axis_x * std::sin(angle / 2.0);
                quat.y = axis_y * std::sin(angle / 2.0);
                quat.z = axis_z * std::sin(angle / 2.0);
                quat.w = std::cos(angle / 2.0);
            }

            pose_msg.pose.orientation = quat;
            pose_pub_->publish(pose_msg);
        }

        
        // 3. TCP Twist
        std::vector<double> tcp_twist = rtde_receive_->getActualTCPSpeed(); // [vx, vy, vz, wx, wy, wz]
        if (tcp_twist.size() == 6) {
            geometry_msgs::msg::TwistStamped twist_msg;
            twist_msg.header.stamp = this->now();
            twist_msg.header.frame_id = "base_link";

            twist_msg.twist.linear.x = tcp_twist[0];
            twist_msg.twist.linear.y = tcp_twist[1];
            twist_msg.twist.linear.z = tcp_twist[2];
            twist_msg.twist.angular.x = tcp_twist[3];
            twist_msg.twist.angular.y = tcp_twist[4];
            twist_msg.twist.angular.z = tcp_twist[5];

            twist_pub_->publish(twist_msg);
        }
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReceiverFromFollowerNode>());
    rclcpp::shutdown();
    return 0;
}

