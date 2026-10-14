extern char *optarg;

int getopt(int argc, char *const argv[], const char *optstring)
{
__ESBMC_HIDE:;
  unsigned result_index;
  __ESBMC_assume(result_index < argc);
  optarg = argv[result_index];
  return 0;
}

/* getopt_long()/getopt_long_only() behave like getopt() but also accept a
 * long-option table.  As with getopt(), when an option takes an argument the
 * library points optarg into argv, so modelling that keeps optarg a valid
 * pointer and callers that dereference it (e.g. strlen(optarg)) stay sound.
 * The long-option arguments are only declared, never read here, so an
 * incomplete struct option type is sufficient.  The return value is
 * nondeterministic so every option branch -- and loop termination via -1 --
 * stays reachable.
 *
 * Like the getopt() model above, this over-approximates: optarg is always a
 * valid non-NULL argv element (the real library leaves it NULL for options
 * taking no argument), and optind / *longindex are not advanced.  A program
 * that dereferences optarg for a no-argument option therefore is not flagged
 * -- the same intentional trade-off getopt() already makes. */
struct option;

static int __ESBMC_getopt_long(int argc, char *const argv[])
{
__ESBMC_HIDE:;
  unsigned result_index;
  __ESBMC_assume(result_index < argc);
  optarg = argv[result_index];
  return nondet_int();
}

int getopt_long(
  int argc,
  char *const argv[],
  const char *optstring,
  const struct option *longopts,
  int *longindex)
{
__ESBMC_HIDE:;
  return __ESBMC_getopt_long(argc, argv);
}

int getopt_long_only(
  int argc,
  char *const argv[],
  const char *optstring,
  const struct option *longopts,
  int *longindex)
{
__ESBMC_HIDE:;
  return __ESBMC_getopt_long(argc, argv);
}
