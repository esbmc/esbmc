#include <assert.h>
int a[4];
int main()
{
  a[0] = 0;
  a[1] = 0;
  a[2] = 7; // witness for the existential
  a[3] = 0;
  assert(__CPROVER_exists{ int i; (i >= 0 && i < 4) && (a[i] == 7) });
  return 0;
}
