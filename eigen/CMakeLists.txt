cmake_minimum_required(VERSION 2.8)

project(BA_eigen)

find_package(Eigen3 REQUIRED)
find_package(glog)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(BA_eigen BA_eigen.cpp)
target_link_libraries(BA_eigen glog::glog Eigen3::Eigen)
