#include <assert.h>
#include <stdio.h>

unsigned long long nondet_ull(void);

int main(void)
{
  unsigned long long x = nondet_ull();
  // %u prints x converted to unsigned int; no value of x is ruled out.
  int r = printf("%u", x);
  assert(x < 10000000000ULL);
  return 0;
}
