#include <assert.h>

int main()
{
  int x = 0;
  _Bool ok = __sync_bool_compare_and_swap(&x, 0, 5);
  assert(ok && x == 5);
  ok = __sync_bool_compare_and_swap(&x, 0, 7);
  assert(!ok && x == 5);

  int v = __sync_val_compare_and_swap(&x, 5, 9);
  assert(v == 5 && x == 9);
  v = __sync_val_compare_and_swap(&x, 5, 1);
  assert(v == 9 && x == 9);

  unsigned char c = 200;
  unsigned char old_c = __sync_val_compare_and_swap(&c, 200, 7);
  assert(old_c == 200 && c == 7);

  long l = 3;
  long old = __sync_lock_test_and_set(&l, 1);
  assert(old == 3 && l == 1);
  __sync_lock_release(&l);
  assert(l == 0);

  int a = 4, b = 6, r = 0;
  __atomic_exchange(&a, &b, &r, __ATOMIC_SEQ_CST);
  assert(a == 6 && b == 6 && r == 4);
  __atomic_exchange(&a, &a, &a, __ATOMIC_SEQ_CST);
  assert(a == 6);
  return 0;
}
