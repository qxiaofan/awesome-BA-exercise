cmake_minimum_required(VERSION 2.8)

project(BA_ceres)

set(CERES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(${CERES_PATH})

find_package(Ceres REQUIRED PATHS "${CERES_PATH}/Thirdparty/ceres-solver")
include_directories(${CERES_INCLUDE_DIRS})

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BA_ceres BA_ceres.cpp)
target_link_libraries(BA_ceres ${CERES_LIBRARIES}
#        ${OpenCV_LIBS}
        )

add_executable(Ceres_exam1_hello_world Ceres_exam1_hello_world.cpp)
target_link_libraries(Ceres_exam1_hello_world ${CERES_LIBRARIES})

add_executable(Ceres_exam2_BAL Ceres_exam2_BAL.cpp)
target_link_libraries(Ceres_exam2_BAL ${CERES_LIBRARIES})

add_executable(Ceres_exam3_helloworld_numeric_diff Ceres_exam3_helloworld_numeric_diff.cpp)
target_link_libraries(Ceres_exam3_helloworld_numeric_diff ${CERES_LIBRARIES})

add_executable(Ceres_exam4_helloworld_analytic_diff Ceres_exam4_helloworld_analytic_diff.cpp)
target_link_libraries(Ceres_exam4_helloworld_analytic_diff ${CERES_LIBRARIES})