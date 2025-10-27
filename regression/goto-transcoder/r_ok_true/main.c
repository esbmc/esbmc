#include<assert.h>
int main() {

  char a[4];
  assert(__CPROVER_r_ok(a, 2));
  return 0;
}
