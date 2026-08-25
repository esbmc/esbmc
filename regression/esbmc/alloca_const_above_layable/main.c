/* R39: a constant alloca above the bound smt_memspace.cpp can lay out made the
   layout constraint unsatisfiable, so every execution -- including the one that
   reaches assert(0) -- was pruned and the program was proved. malloc's constant
   request is classified (#6660); alloca was left out of that classification. */
#include <assert.h>

int main(void)
{
  char *p = __builtin_alloca(-1);

  p[0] = 1;
  assert(0);
  return 0;
}
