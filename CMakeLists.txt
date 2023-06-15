cmake_minimum_required(VERSION 3.0.0)
project(tracker VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV 4.2 REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp rosbag sensor_msgs cv_bridge)
find_package(Eigen3 REQUIRED)

set(COMPILE_OPTIONS -Wall -Wshadow -ffast-math -march=native -std=c++17 -fPIC)

add_library(gyro_lk STATIC
    src/utils.cpp
    src/gyro_lk.cpp
    src/load_data.cpp
    include/utils.hpp
    include/gyro_lk.hpp
    include/load_data.hpp
)

target_link_libraries(gyro_lk PUBLIC
    ${OpenCV_LIBRARIES} 
    ${catkin_LIBRARIES}
    ${Eigen3_LIBRARIES}
)

target_include_directories(gyro_lk PUBLIC
    include
)

target_include_directories(gyro_lk SYSTEM PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)
target_compile_options(gyro_lk PRIVATE ${COMPILE_OPTIONS})

add_executable(tracker src/main.cpp)
target_compile_options(tracker PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(tracker PRIVATE gyro_lk)
target_compile_definitions(gyro_lk PUBLIC SOURCE_DIR="${PROJECT_SOURCE_DIR}/")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
