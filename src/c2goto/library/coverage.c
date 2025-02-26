#include <stdlib.h>

void __ESBMC_malloc_argv(int argc, char **argv)
{
__ESBMC_HIDE:;
  char **argv_copy = malloc((argc + 1) * sizeof(char *));
  for (int i = 0; i <= argc; i++)
  {
      argv_copy[i] =__ESBMC_inf_str;
  }
  argv = argv_copy;
}