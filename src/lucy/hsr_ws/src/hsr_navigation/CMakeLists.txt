cmake_minimum_required(VERSION 3.0.2)
project(hsr_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    visualization_msgs
    tf2_ros
    tf2_geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/navigation_manager.py
  scripts/a_star_planner.py
  scripts/waypoints_extractor.py
  scripts/waypoints_follower.py
  scripts/speech_recognition_node.py
  scripts/dialog_manager.py
  scripts/field_planner.py



  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)