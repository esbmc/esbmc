#include <assert.h>
#include <stdlib.h>
int main() {
  char a[4];
  assert(__CPROVER_r_ok(a, 4));
  assert(!__CPROVER_r_ok(a, 8));
  assert(__CPROVER_r_ok(a + 2, 2));
  assert(!__CPROVER_r_ok(a + 2, 3));
  assert(!__CPROVER_r_ok((void *)0, 1));
  assert(__CPROVER_OBJECT_SIZE(a) == 4);
  assert(__CPROVER_OBJECT_SIZE((void *)0) == 0);
  assert(__CPROVER_same_object(a, a + 1));
  assert(__CPROVER_POINTER_OFFSET(a + 3) == 3);
  assert(!__CPROVER_DYNAMIC_OBJECT(a));
  assert(__CPROVER_LIVE_OBJECT(a));
  char *d = malloc(8);
  assert(__CPROVER_DYNAMIC_OBJECT(d));
  assert(__CPROVER_w_ok(d, 8));
  assert(!__CPROVER_rw_ok(d, 9));
  free(d);
  assert(!__CPROVER_LIVE_OBJECT(d));
  assert(!__CPROVER_r_ok(d, 1));
  return 0;
}
