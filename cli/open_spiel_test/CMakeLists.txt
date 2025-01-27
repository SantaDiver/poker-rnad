cmake_minimum_required(VERSION 3.17.0)

include(CTest)
enable_testing()

set(TARGET_NAME open_spiel_test)
add_executable(${TARGET_NAME} main.cpp)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/RNadBot.cc)

target_include_directories(${TARGET_NAME} PRIVATE ${poker-rnad_SOURCE_DIR}/contrib/open_spiel)
target_include_directories(${TARGET_NAME} PRIVATE ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/abseil-cpp)
target_link_libraries(
    ${TARGET_NAME}
    ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/build/libopen_spiel.dylib
)

list(APPEND CMAKE_PREFIX_PATH "${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/libtorch/libtorch")
find_package(Torch REQUIRED)
target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
