#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int **g;

void free_g(void)
{
  g = 0;
}

int main()
{
  g = malloc(sizeof(int *));
  if (!g)
    return 0;
  *g = malloc(sizeof(int));
  atexit(free_g);
  if (__VERIFIER_nondet_int())
    exit(1);
  free(*g);
  return 0;
}
