// #10135: a compound assignment to a bit-field wider than int runs in the
// computation type and stores the result truncated to the field's width.
#include <assert.h>

extern unsigned long __VERIFIER_nondet_ulong(void);

struct s
{
  unsigned long f : 52;
  long g : 40;
};

int main(void)
{
  struct s p;
  p.f = __VERIFIER_nondet_ulong();
  unsigned long m = __VERIFIER_nondet_ulong();
  unsigned long before = p.f;
  p.f &= m;
  assert(p.f == (before & m));

  p.f = 0xFFFFFFFFFFFFFUL;
  p.f += 1;
  assert(p.f == 0);

  p.f -= 1;
  assert(p.f == 0xFFFFFFFFFFFFFUL);

  p.f ^= ~0UL;
  assert(p.f == 0);

  p.f |= 1UL << 51;
  p.f *= 2;
  assert(p.f == 0);

  p.g = -3;
  p.g += 1UL;
  assert(p.g == -2);

  p.f = 10;
  p.f += 1.5;
  p.g *= 2.5;
  assert(p.f == 11 && p.g == -5);

  unsigned _BitInt(52) x = 0xFFFFFFFFFFFFFUL;
  x &= m;
  assert(x == (m & 0xFFFFFFFFFFFFFUL));
  return 0;
}
