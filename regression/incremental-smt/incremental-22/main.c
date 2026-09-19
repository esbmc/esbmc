#include <assert.h>
int nondet_int();
int main()
{
  int a = nondet_int();
  int b = nondet_int();
  int *c = &a;
  int *d = &b;
  // The inductive step havocs a and b, which the loop writes through c and d,
  // so only a property that holds for every a > 0 is provable.
  while(a > 0)
  {
    *c = *c - 1;
    *d = *d + 1;
    assert(*c >= 0);
  }
  return 1;
}
