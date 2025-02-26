#include <stdio.h>
#include <stdlib.h>

// for argc & argv
__attribute__((annotate("__ESBMC_inf_size"))) char __ESBMC_inf_str[1];
char **ESBMC_malloc_argv(int argc)
{
__ESBMC_HIDE:;
  char **argv_copy = (char **)malloc((argc + 1) * sizeof(char *));
  for (int i = 0; i <= argc; i++)
  {
    argv_copy[i] = __ESBMC_inf_str;
  }
  return argv_copy;
}