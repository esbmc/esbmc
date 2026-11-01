#include <assert.h>
#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int main()
{
  /* `unused` and `i` are uninitialised locals and `errno` is an uninitialised
   * global of ESBMC's libc model. ESBMC models all three as nondet, but none
   * is a program input, so testcase.xml must hold only the values of the two
   * __VERIFIER_nondet_int() calls, in call order. */
  int unused, i, x = __VERIFIER_nondet_int();
  int *p = malloc(sizeof(int));
  int y = __VERIFIER_nondet_int();
  if (x == 42 && y == 7)
    assert(0);
  free(p);
  return 0;
}
