add_subdirectory(open_spiel/pybind11)

add_library(thread-pool INTERFACE)
target_include_directories(thread-pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include)

add_subdirectory(libfork)
# add_library(libfork INTERFACE)
# target_include_directories(libfork INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/libfork/include)