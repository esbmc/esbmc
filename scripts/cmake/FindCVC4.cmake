find_package(CVC4 HINTS ${CAMADA_CVC4_DIR}/lib/cmake/CVC4 $ENV{HOME}/cvc4)

if(CVC4_FOUND)
  # Remove any suffix from CVC4's version string
  string(REGEX REPLACE "([0-9]\\.[0-9]).*" "\\1" CVC4_VERSION "${CVC4_VERSION}")

  set(CVC4_MIN_VERSION "1.8")
  if(CVC4_VERSION VERSION_LESS CVC4_MIN_VERSION)
    message(FATAL_ERROR "Expected version ${CVC4_MIN_VERSION} or greater")
  endif()

  # Search for symfpu headers and set it a CVC4 include
  find_path(CAMADA_CVC4_SYMFPU_DIR symfpu/core/unpackedFloat.h PATHS)

  # TODO: this symfpu check should be clever: we should check if cvc4 was
  # actually built with symfpu and then abort/warn the user
endif()
