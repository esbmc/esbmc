/* #7521 field-sensitivity corpus: a local struct, 8 field writes x 200 cycles.
 * The struct has 512 fields; F<n>(p) declares n padding fields. */
#include <assert.h>

int nondet_int(void);

#define F1(p) int p;
#define F2(p) F1(p##0) F1(p##1)
#define F4(p) F2(p##0) F2(p##1)
#define F8(p) F4(p##0) F4(p##1)
#define F16(p) F8(p##0) F8(p##1)
#define F32(p) F16(p##0) F16(p##1)
#define F64(p) F32(p##0) F32(p##1)
#define F128(p) F64(p##0) F64(p##1)
#define F256(p) F128(p##0) F128(p##1)
struct S
{
  int a0, a1, a2, a3, a4, a5, a6, a7;
  F256(p) F128(q) F64(r) F32(s) F16(t) F8(u)
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
