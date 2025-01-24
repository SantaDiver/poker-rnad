cmake_minimum_required(VERSION 3.17.0)

include(CTest)
enable_testing()

set(TARGET_NAME open_spiel_test)
add_executable(${TARGET_NAME} main.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${poker-rnad_SOURCE_DIR}/contrib/open_spiel)
target_include_directories(${TARGET_NAME} PRIVATE ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/abseil-cpp)
target_link_libraries(
    open_spiel_test
    ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/build/libopen_spiel.dylib
)
