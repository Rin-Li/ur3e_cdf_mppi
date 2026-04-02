cmake_minimum_required(VERSION 3.14)
project(sender_to_follower)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ur_rtde REQUIRED)

add_executable(servo_j_to_follower_node src/servo_j_to_follower_node.cpp)

target_link_libraries(servo_j_to_follower_node
  rtde
)

ament_target_dependencies(servo_j_to_follower_node
  rclcpp
  trajectory_msgs
  ur_rtde
)

install(TARGETS
  servo_j_to_follower_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
