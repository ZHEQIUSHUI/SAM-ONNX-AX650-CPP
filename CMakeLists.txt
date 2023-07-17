# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# check if building type is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/axpi_bsp_sdk/msp/out)
endif()

message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(SegmentAnything-AX650)

# 添加 FLAGS 检查代码是否有明显 bug
include(cmake/overlook.cmake)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-aarch64-none-gnu-1.16.0)
endif()
message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories("${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_LIB onnxruntime)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(axpi_pro_bsp_sdk/msp/out/include)
link_directories(axpi_pro_bsp_sdk/msp/out/lib)

add_library(sam STATIC
    ${CMAKE_SOURCE_DIR}/src/Runner/BaseRunner.cpp
    ${CMAKE_SOURCE_DIR}/src/Runner/ax_model_runner_ax650.cpp
)

add_executable(main ${CMAKE_SOURCE_DIR}/src/main.cpp)


target_link_libraries(main sam)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main gomp ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})

target_link_libraries(main ax_engine ax_interpreter ax_sys ax_ivps)

install(TARGETS main DESTINATION bin)