add_subdirectory(tuple)
add_subdirectory(fp)

add_library(smt array_conv.cpp smt_byteops.cpp smt_casts.cpp smt_conv.cpp smt_memspace.cpp smt_overflow.cpp)
target_include_directories(smt
    PRIVATE ${Boost_INCLUDE_DIRS}
)