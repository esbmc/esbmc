#include <assert.h>
int a[4];
int main()
{
  for (int k = 0; k < 4; k++)
    a[k] = 0;
  // __CPROVER_forall lowers to a "forall" quantifier irep; every element is 0.
  assert(__CPROVER_forall{ int i; (i >= 0 && i < 4) ==> (a[i] == 0) });
  return 0;
}
