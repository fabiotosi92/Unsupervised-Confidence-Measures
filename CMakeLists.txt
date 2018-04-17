cmake_minimum_required( VERSION 2.8 )
project( bmcv2017 )

set( OpenCV_DIR "/usr/share/OpenCV/" )
find_package( OpenCV REQUIRED )

include_directories(include $(OpenCV_INCLUDE_DIRS))
file(GLOB SOURCES src/*.cpp)
add_executable( bmcv2017 ${SOURCES})
target_link_libraries( bmcv2017 ${OpenCV_LIBS} )
