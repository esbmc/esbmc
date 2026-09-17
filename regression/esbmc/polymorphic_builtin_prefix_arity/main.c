#include <assert.h>

/* The overflow and carry arms index arguments[0..2] and [0..3] directly, but
 * the name prefix they match on is not reserved to the builtin, so a user
 * function reaches them with whatever arity it was declared with -- and there
 * is then nothing to index. */
int __builtin_add_overflow_mine(int *p);
int __builtin_addc_mine(int *p, int *q);

int main(void)
{
  int x = 0;
  int a = __builtin_add_overflow_mine(&x);
  int b = __builtin_addc_mine(&x, &x);
  assert(a == a && b == b);
  return 0;
}
