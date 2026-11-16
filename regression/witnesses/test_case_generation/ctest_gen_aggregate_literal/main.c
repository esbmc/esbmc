#include <assert.h>

extern int __VERIFIER_nondet_int(void);

struct pair
{
  int a, b;
};

struct pair unused, used;

union word
{
  int i;
  float f;
} w;

int main()
{
  /* Constant propagation folds each write into a struct literal holding the
   * nondet value, e.g. `used == { .a=nondet4, .b=nondet3 }`. Every call is
   * still a harness input, in call order, including the three the property
   * never reads. */
  w.i = __VERIFIER_nondet_int();
  unused.a = __VERIFIER_nondet_int();
  unused.b = __VERIFIER_nondet_int();
  used.b = __VERIFIER_nondet_int();
  used.a = __VERIFIER_nondet_int();
  if (used.a == 1 && used.b == 2)
    assert(0);
  return 0;
}
