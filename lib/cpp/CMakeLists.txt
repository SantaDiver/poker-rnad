set(LIB_NAME poker_rnad_cpp)
add_library(${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ActorWorker.cpp)

target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel)
target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/abseil-cpp)
target_include_directories(${LIB_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

# [TODO] fix this hack
list(APPEND CMAKE_PREFIX_PATH "/Users/penguin-diver/miniconda3/lib/python3.12/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_library(open_spiel SHARED IMPORTED)
set_target_properties(open_spiel PROPERTIES
    # IMPORTED_LOCATION "/Users/penguin-diver/miniconda3/lib/python3.12/site-packages"
    IMPORTED_LOCATION "/opt/conda/lib/python3.11/site-packages/pyspiel.so"
)

target_link_libraries(${LIB_NAME} PUBLIC
    ${TORCH_LIBRARIES}
    open_spiel
    thread-pool
)
