/* #7521 field-sensitivity corpus: a global struct written through a pointer in a callee.
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

struct S db;

void cycle(struct S *d)
{
  d->a1 = d->a0 + 1;
  d->a2 = d->a1 + 1;
  d->a3 = d->a2 + 1;
  d->a4 = d->a3 + 1;
  d->a5 = d->a4 + 1;
  d->a6 = d->a5 + 1;
  d->a7 = d->a6 + 1;
  d->a0 = d->a7 - 6;
}

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 1000);
  db.a0 = x;
  for (int c = 0; c < 200; c++)
    cycle(&db);
  assert(db.a0 == x + 199);
  return 0;
}
