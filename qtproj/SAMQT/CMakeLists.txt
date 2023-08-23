cmake_minimum_required(VERSION 3.5)

project(SAMQT LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libsam)

find_package(Qt6 COMPONENTS Widgets)
if(Qt6_FOUND)
  set(QTWidget Qt6::Widgets)
else()
  find_package(Qt5 COMPONENTS Widgets REQUIRED)
  set(QTWidget Qt5::Widgets)
endif()


if(ANDROID)
  add_library(SAMQT SHARED
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
  )
else()
  add_executable(SAMQT
    main.cpp
    mainwindow.cpp
    mainwindow.h
    myqlabel.h
    mainwindow.ui
    style/DarkStyle.cpp
    style/DarkStyle.h
  )
endif()

target_link_libraries(SAMQT PRIVATE ${QTWidget})
include_directories(style)
include_directories(${ONNXRUNTIME_DIR}/include)
target_link_directories(SAMQT PRIVATE ${ONNXRUNTIME_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(SAMQT PRIVATE sam)
target_link_libraries(SAMQT PRIVATE ${OpenCV_LIBS})
target_link_libraries(SAMQT PRIVATE gomp onnxruntime)

if(BUILD_WITH_AX650)
    target_link_directories(SAMQT PRIVATE ${BSP_MSP_DIR}/lib)
    target_link_libraries(SAMQT PRIVATE ax_engine ax_interpreter ax_sys ax_ivps)
endif()
