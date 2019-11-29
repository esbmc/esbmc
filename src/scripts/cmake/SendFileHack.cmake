
# Configure if sendfile exists (used on ./esbmc/esbmc_parseoptions.cpp)

include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS(sendfile /usr/include/sys/sendfile.h HAVE_SENDFILE)
if ((DEFINED HAVE_SENDFILE) AND (HAVE_SENDFILE STREQUAL 1))
  message(STATUS "sendfile found, esbmc_parseoptions will use it")
  set(HAVE_SENDFILE_ESBMC 1)
else()
    message(STATUS "sendfile not found!")
  set(HAVE_SENDFILE_ESBMC 0)
endif()