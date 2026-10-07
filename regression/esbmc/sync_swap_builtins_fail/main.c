#include <assert.h>

int main()
{
  int x = 0, y = 0;
  long l = 3, m = 3;
  int a = 4, b = 6, r = 0;

  __sync_bool_compare_and_swap(&x, 0, 5);
  __sync_val_compare_and_swap(&y, 0, 5);
  __sync_lock_test_and_set(&l, 1);
  __sync_lock_release(&m);
  __atomic_exchange(&a, &b, &r, __ATOMIC_SEQ_CST);

  assert(x == 0 || y == 0 || l == 3 || m == 3 || a == 4 || r == 0);
  return 0;
}
