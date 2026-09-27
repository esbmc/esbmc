#include <assert.h>

char *nondet_charp(void);
char b[64];

/* A nondet pointer pinned to b by an assume: its value set is unknown. */
int main(void)
{
  char *p = nondet_charp();
  __ESBMC_assume(p == &b[0]);
  *p = 3;
  assert(b[0] == 0);
  return 0;
}
