/* #7521 field-sensitivity corpus: a global struct written through a pointer in a callee.
 * The struct has 8 fields; F<n>(p) declares n padding fields. */
#include <assert.h>

int nondet_int(void);

struct S
{
  int a0, a1, a2, a3, a4, a5, a6, a7;
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
  assert(db.a0 == x + 200);
  return 0;
}
