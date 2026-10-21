#include <stdlib.h>
void g(int **pp)
{
  __ESBMC_requires(1);
  __ESBMC_assigns(*pp);
  __ESBMC_frees(*pp);
  __ESBMC_ensures(__ESBMC_is_fresh(*pp, sizeof(int)));
  free(*pp);
  *pp = malloc(sizeof(int));
}
int main()
{
  int *p = malloc(sizeof(int));
  int *q = p;
  g(&p);
#ifdef UAF
  int r = *q;
#endif
  free(p);
}
