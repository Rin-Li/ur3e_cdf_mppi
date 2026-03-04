## TEST
https://prod.liveshare.vsengsaas.visualstudio.com/join?1D577F85C40CE46BE014F5D7BC3A0D8D9EB3

kklab@kklab-DAIV-DGZ520:~/ur_ws$ colcon build --packages-select sender_to_follower --event-handlers console_direct+
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/receiver_from_follower' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/leader' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/flag_publisher' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/action_bridge' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/custom_msgs' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/clutch' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.ament:The path '/home/kklab/ur_ws/install/camera_publisher' in the environment variable AMENT_PREFIX_PATH doesn't exist
[0.353s] WARNING:colcon.colcon_ros.prefix_path.catkin:The path '/home/kklab/ur_ws/install/receiver_from_follower' in the environment variable CMAKE_PREFIX_PATH doesn't exist
[0.354s] WARNING:colcon.colcon_ros.prefix_path.catkin:The path '/home/kklab/ur_ws/install/leader' in the environment variable CMAKE_PREFIX_PATH doesn't exist
[0.354s] WARNING:colcon.colcon_ros.prefix_path.catkin:The path '/home/kklab/ur_ws/install/action_bridge' in the environment variable CMAKE_PREFIX_PATH doesn't exist
[0.354s] WARNING:colcon.colcon_ros.prefix_path.catkin:The path '/home/kklab/ur_ws/install/custom_msgs' in the environment variable CMAKE_PREFIX_PATH doesn't exist
Starting >>> sender_to_follower
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0        
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done                
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done              
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found ament_cmake: 0.9.12 (/opt/ros/foxy/share/ament_cmake/cmake)
-- Found PythonInterp: /usr/bin/python3 (found suitable version "3.8.10", minimum required is "3") 
-- Using PYTHON_EXECUTABLE: /usr/bin/python3
-- Found rclcpp: 2.4.3 (/opt/ros/foxy/share/rclcpp/cmake)
-- Using all available rosidl_typesupport_c: rosidl_typesupport_fastrtps_c;rosidl_typesupport_introspection_c
-- Found rosidl_adapter: 1.3.1 (/opt/ros/foxy/share/rosidl_adapter/cmake)
-- Found OpenSSL: /usr/lib/x86_64-linux-gnu/libcrypto.so (found version "1.1.1f")  
-- Found FastRTPS: /opt/ros/foxy/include  
-- Using all available rosidl_typesupport_cpp: rosidl_typesupport_fastrtps_cpp;rosidl_typesupport_introspection_cpp
-- Found rmw_implementation_cmake: 1.0.4 (/opt/ros/foxy/share/rmw_implementation_cmake/cmake)
-- Using RMW implementation 'rmw_fastrtps_cpp' as default
-- Looking for pthread.h
-- Looking for pthread.h - found                       
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed    
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found       
-- Found Threads: TRUE  
-- Found sensor_msgs: 2.0.5 (/opt/ros/foxy/share/sensor_msgs/cmake)
-- Found trajectory_msgs: 2.0.5 (/opt/ros/foxy/share/trajectory_msgs/cmake)
-- Found tf2: 0.13.14 (/opt/ros/foxy/share/tf2/cmake)  
-- Found tf2_ros: 0.13.14 (/opt/ros/foxy/share/tf2_ros/cmake)
-- Found tf2_geometry_msgs: 0.13.14 (/opt/ros/foxy/share/tf2_geometry_msgs/cmake)
-- Found eigen3_cmake_module: 0.1.1 (/opt/ros/foxy/share/eigen3_cmake_module/cmake)
-- Found Eigen3: TRUE (found version "3.3.7") 
-- Found Boost: /usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0/BoostConfig.cmake (found version "1.71.0") found components: system thread 
-- Configuring done                                    
-- Generating done
-- Build files have been written to: /home/kklab/ur_ws/build/sender_to_follower
Scanning dependencies of target ur3_inverse_kinematics_lib
Scanning dependencies of target delta_vector_camera_to_base_node
[ 15%] Building CXX object CMakeFiles/ur3_inverse_kinematics_lib.dir/src/ur3_forward_kinematics.cpp.o
[ 15%] Building CXX object CMakeFiles/ur3_inverse_kinematics_lib.dir/src/ur3_inverse_kinematics.cpp.o
[ 23%] Building CXX object CMakeFiles/delta_vector_camera_to_base_node.dir/src/delta_vector_camera_to_base_node.cpp.o
/home/kklab/ur_ws/src/sender_to_follower/src/ur3_forward_kinematics.cpp: In function ‘Eigen::Vector3d ur3_ik::ur3_forward_kinematics(const std::array<double, 6>&, const Vector3d&)’:
/home/kklab/ur_ws/src/sender_to_follower/src/ur3_forward_kinematics.cpp:26:18: warning: unused variable ‘q5’ [-Wunused-variable]
   26 |     const double q5 = q[4];
      |                  ^~
/home/kklab/ur_ws/src/sender_to_follower/src/ur3_forward_kinematics.cpp:27:18: warning: unused variable ‘q6’ [-Wunused-variable]
   27 |     const double q6 = q[5];
      |                  ^~
[ 30%] Linking CXX static library libur3_inverse_kinematics_lib.a
[ 30%] Built target ur3_inverse_kinematics_lib
Scanning dependencies of target ik_anglevel_node
Scanning dependencies of target send_jointvel_to_follower_node
Scanning dependencies of target sender_to_follower_node
Scanning dependencies of target ik_fixpoint_node
[ 46%] Building CXX object CMakeFiles/send_jointvel_to_follower_node.dir/src/send_jointvel_to_follower_node.cpp.o
[ 46%] Building CXX object CMakeFiles/ik_anglevel_node.dir/src/ik_anglevel_node.cpp.o
[ 53%] Building CXX object CMakeFiles/ik_fixpoint_node.dir/src/ik_fixpoint_node.cpp.o
[ 61%] Building CXX object CMakeFiles/sender_to_follower_node.dir/src/sender_to_follower_node.cpp.o
[ 69%] Linking CXX executable delta_vector_camera_to_base_node
[ 69%] Built target delta_vector_camera_to_base_node       
[ 76%] Linking CXX executable sender_to_follower_node      
[ 76%] Built target sender_to_follower_node                
[ 84%] Linking CXX executable ik_fixpoint_node               
[ 84%] Built target ik_fixpoint_node                         
[ 92%] Linking CXX executable send_jointvel_to_follower_node 
[ 92%] Built target send_jointvel_to_follower_node           
[100%] Linking CXX executable ik_anglevel_node               
[100%] Built target ik_anglevel_node                          
-- Install configuration: ""
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/sender_to_follower_node
-- Set runtime path of "/home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/sender_to_follower_node" to ""
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/send_jointvel_to_follower_node
-- Set runtime path of "/home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/send_jointvel_to_follower_node" to ""
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/ik_fixpoint_node
-- Set runtime path of "/home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/ik_fixpoint_node" to ""
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/ik_anglevel_node
-- Set runtime path of "/home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/ik_anglevel_node" to ""
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/delta_vector_camera_to_base_node
-- Set runtime path of "/home/kklab/ur_ws/install/sender_to_follower/lib/sender_to_follower/delta_vector_camera_to_base_node" to ""
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/include/sender_to_follower
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/include/sender_to_follower/ur3_inverse_kinematics.hpp
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/urdf
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/urdf/ur3e.urdf
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/ament_index/resource_index/package_run_dependencies/sender_to_follower
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/ament_index/resource_index/parent_prefix_path/sender_to_follower
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/environment/ament_prefix_path.sh
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/environment/ament_prefix_path.dsv
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/environment/path.sh
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/environment/path.dsv
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/local_setup.bash
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/local_setup.sh
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/local_setup.zsh
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/local_setup.dsv
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/package.dsv
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/ament_index/resource_index/packages/sender_to_follower
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/cmake/sender_to_followerConfig.cmake
-- Installing: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/cmake/sender_to_followerConfig-version.cmake
-- Up-to-date: /home/kklab/ur_ws/install/sender_to_follower/share/sender_to_follower/package.xml
--- stderr: sender_to_follower
/home/kklab/ur_ws/src/sender_to_follower/src/ur3_forward_kinematics.cpp: In function ‘Eigen::Vector3d ur3_ik::ur3_forward_kinematics(const std::array<double, 6>&, const Vector3d&)’:
/home/kklab/ur_ws/src/sender_to_follower/src/ur3_forward_kinematics.cpp:26:18: warning: unused variable ‘q5’ [-Wunused-variable]
   26 |     const double q5 = q[4];
      |                  ^~
/home/kklab/ur_ws/src/sender_to_follower/src/ur3_forward_kinematics.cpp:27:18: warning: unused variable ‘q6’ [-Wunused-variable]
   27 |     const double q6 = q[5];
      |                  ^~
---
Finished <<< sender_to_follower [11.8s]

Summary: 1 package finished [12.0s]
  1 package had stderr output: sender_to_follower
kklab@kklab-DAIV-DGZ520:~/ur_ws$ 
