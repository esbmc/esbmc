/* FUNCTION: getopt */

extern char *optarg;

inline int getopt(int argc, char * const argv[],
                  const char *optstring)
{
  __ESBMC_HIDE:;
  unsigned result_index;
  __ESBMC_assume(result_index<argc);
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(optstring),
    "getopt zero-termination of 3rd argument");
  #endif
  optarg = argv[result_index];
}
