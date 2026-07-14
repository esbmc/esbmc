/* #5565 boundary: the getopt_long model points optarg into argv, a *bounded*
 * real object -- it does not over-approximate optarg to an unbounded buffer.
 * Reading far past the argv element must therefore still be caught as an
 * out-of-bounds access (this would spuriously pass if the model invented an
 * infinite buffer). */
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

int main()
{
  int argc = 2;
  char **argv = malloc((argc + 1) * sizeof(char *));
  argv[argc] = 0;
  for (int i = 0; i < argc; i++)
  {
    argv[i] = malloc(4); /* 4-byte argv element */
    argv[i][0] = 0;
  }

  int c = getopt_long(argc, argv, "a", (struct option *)0, 0);
  if (c == 'a')
    optarg[100] =
      'x'; /* out of bounds for the 4-byte object optarg points to */

  for (int i = 0; i < argc; i++)
    free(argv[i]);
  free(argv);
  return 0;
}
