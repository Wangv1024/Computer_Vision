cmake_minimum_required(VERSION 3.5)
project(project1_1a)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(wink DetectWink.cc)
add_executable(silence DetectSilence.cc)


target_link_libraries( wink ${OpenCV_LIBS} )
target_link_libraries( silence ${OpenCV_LIBS} )

add_custom_command(TARGET silence POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Mouth.xml $<TARGET_FILE_DIR:silence>)

add_custom_command(TARGET wink POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/wingimages $<TARGET_FILE_DIR:wink>/wingimages)

add_custom_command(TARGET silence POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/silenceimages $<TARGET_FILE_DIR:silence>/silenceimages)