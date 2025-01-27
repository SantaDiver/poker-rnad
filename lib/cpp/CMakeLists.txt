cmake_minimum_required(VERSION 3.17.0)

set(LIB_NAME poker-rnad)
add_library(${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/RNadBot.cc)

target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel)
target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/abseil-cpp)
target_link_libraries(
    ${LIB_NAME}
    ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/build/libopen_spiel.dylib
)

# list(APPEND CMAKE_PREFIX_PATH "${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/libtorch/libtorch")
# find_package(Torch REQUIRED)
# target_include_directories(${LIB_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
# target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")
