/* #7521 field-sensitivity corpus: a local struct, 8 field writes x 200 cycles.
 * The struct has 8 fields; F<n>(p) declares n padding fields. */
#include <assert.h>

int nondet_int(void);

struct S
{
  int a0, a1, a2, a3, a4, a5, a6, a7;
};

int main(void)
{
  struct S s;
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 1000);
  s.a0 = x;
  for (int c = 0; c < 200; c++)
  {
    s.a1 = s.a0 + 1;
    s.a2 = s.a1 + 1;
    s.a3 = s.a2 + 1;
    s.a4 = s.a3 + 1;
    s.a5 = s.a4 + 1;
    s.a6 = s.a5 + 1;
    s.a7 = s.a6 + 1;
    s.a0 = s.a7 - 6;
  }
  assert(s.a0 == x + 200);
  return 0;
}
