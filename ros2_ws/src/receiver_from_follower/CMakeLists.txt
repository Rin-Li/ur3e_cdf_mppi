cmake_minimum_required(VERSION 3.8)
project(receiver_from_follower)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ur_rtde REQUIRED)

add_executable(receiver_from_follower_node src/receiver_from_follower_node.cpp)

target_link_libraries(receiver_from_follower_node
  rtde
)

ament_target_dependencies(receiver_from_follower_node
  rclcpp
  sensor_msgs
  geometry_msgs
  ur_rtde
  custom_msgs
)

install(TARGETS receiver_from_follower_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
