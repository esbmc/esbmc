#include <assert.h>
int a[4];
int main()
{
  a[0] = 0;
  a[1] = 5; // breaks the forall
  a[2] = 0;
  a[3] = 0;
  assert(__CPROVER_forall{ int i; (i >= 0 && i < 4) ==> (a[i] == 0) });
  return 0;
}
