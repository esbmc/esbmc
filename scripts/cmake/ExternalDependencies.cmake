# Module to add dependencies that do not belong
# anywhere else

include(FetchContent)
# FMT
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 7.1.3)
FetchContent_MakeAvailable(fmt)