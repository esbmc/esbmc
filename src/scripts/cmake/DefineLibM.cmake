# Sets LIBM

if(ENABLE_LIBM)
  set(LIBM_PATTERN "${PROJECT_SOURCE_DIR}/c2goto/library/libm/*.c")
else()
  set(LIBM_PATTERN "")
endif()

message(STATUS "LIBM: ${LIBM_PATTERN}")