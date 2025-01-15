cmake_minimum_required(VERSION 3.8)
project(grp_pibot25)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)


# Python scripts
install( PROGRAMS playground/test_move DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/scan_echo DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight_sim DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight_sim_auto DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight_auto DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/detection_bottle_publisher DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/MapExplorer DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/RRTExplorer DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight_auto_Astar DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight_sim_Astar DESTINATION lib/${PROJECT_NAME} )
install( PROGRAMS playground/straight_nav2 DESTINATION lib/${PROJECT_NAME} )
