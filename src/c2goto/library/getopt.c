extern char *optarg;

int getopt(int argc, char *const argv[], const char *optstring)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  unsigned result_index;
  __ESBMC_assume(result_index < argc);
  optarg = argv[result_index];
  return 0;
}
