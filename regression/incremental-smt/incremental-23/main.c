#include <assert.h>
int nondet_int();
int main()
{
  int a = nondet_int();
  int b = nondet_int();
  int sum = a + b;
  int *c = &a;
  int *d = &b;
  while(a > 0)
  {
    *c = *c - 1;
    *d = *d + 1;
  }
  int sum2 = *c + *d;
  assert(sum != sum2);
  return 1;
}
