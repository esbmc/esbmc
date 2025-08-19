## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME ESBMC test)
set(CTEST_NIGHTLY_START_TIME 01:00:00 UTC)

if(CMAKE_VERSION VERSION_GREATER 3.14)
  set(CTEST_SUBMIT_URL https://my.cdash.org/submit.php?project=ESBMC+test)
else()
  set(CTEST_DROP_METHOD "https")
  set(CTEST_DROP_SITE "my.cdash.org")
  set(CTEST_DROP_LOCATION "/submit.php?project=ESBMC+test")
endif()

set(CTEST_TLS_VERIFY ON)        # requires CMake/CTest ≥ 3.30
set(CTEST_TLS_VERSION "1.2")    # minimum TLS; 1.2 is the default since 3.31
set(CTEST_SUBMIT_RETRY_COUNT 3)
set(CTEST_SUBMIT_RETRY_DELAY 10)
set(CTEST_DROP_SITE_CDASH TRUE)