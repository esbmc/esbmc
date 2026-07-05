/* #5565: getopt_long() must leave optarg pointing into argv (a valid object),
 * mirroring getopt().  Without an operational model getopt_long had no body,
 * so optarg stayed an unconstrained/NULL pointer and dereferencing it
 * (if (*optarg) / strlen(optarg)) raised a spurious "invalid pointer" failure
 * -- the residual comm_3args_ok false alarm after the #5564 memory-leak fix. */
#include <stdlib.h>

extern char *optarg;

struct option
{
  const char *name;
  int has_arg;
  int *flag;
  int val;
};

int getopt_long(
  int argc,
  char *const argv[],
  const char *optstring,
  const struct option *longopts,
  int *longindex);

int getopt_long_only(
  int argc,
  char *const argv[],
  const char *optstring,
  const struct option *longopts,
  int *longindex);

size_t strlen(const char *s);

int main()
{
  int argc = 3;
  char **argv = malloc((argc + 1) * sizeof(char *));
  argv[argc] = 0;
  for (int i = 0; i < argc; i++)
  {
    argv[i] = malloc(4);
    argv[i][0] = 'a';
    argv[i][1] = 0;
  }

  struct option lo[1] = {{0, 0, 0, 0}};
  int c = getopt_long(argc, argv, "abc", lo, 0);
  if (c == -1)
    c = getopt_long_only(argc, argv, "abc", lo, 0);

  /* getopt_long returns a nondet option char; on any branch that consumes an
   * argument the program dereferences optarg.  This must be sound. */
  if (c == 'd')
    if (*optarg)
    {
      size_t n = strlen(optarg);
      (void)n;
    }

  for (int i = 0; i < argc; i++)
    free(argv[i]);
  free(argv);
  return 0;
}
