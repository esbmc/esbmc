add_library(smtfp fp_conv.cpp)
target_include_directories(smtfp
    PRIVATE ${Boost_INCLUDE_DIRS}
)