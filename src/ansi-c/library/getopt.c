#include "intrinsics.h"

extern char *optarg;

int getopt(int argc, char * const argv[], const char *optstring)
{
  __ESBMC_HIDE:;
  unsigned result_index;
  __ESBMC_assume(result_index<argc);
  optarg = argv[result_index];
}

int getopt_strabs(int argc, char * const argv[], const char *optstring)
{
  __ESBMC_HIDE:;
  unsigned result_index;
  __ESBMC_assume(result_index<argc);
  __ESBMC_assert(__ESBMC_is_zero_string(optstring),
    "getopt zero-termination of 3rd argument");
  optarg = argv[result_index];
}
