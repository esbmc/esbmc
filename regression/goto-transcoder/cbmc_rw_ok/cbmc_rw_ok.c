#include<assert.h>
int main() {
  int x = 5;
  int *p = &x;
  __CPROVER_assert(__CPROVER_rw_ok(p, sizeof(int)), "rw ok");
  *p = 10;
  assert(x == 10);
  return 0;
}
