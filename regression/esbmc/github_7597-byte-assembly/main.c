/* #7597's shape with the written value assembled rather than copied: the
 * counter is a member of the struct whose array member takes one write per
 * iteration, and each write is `(hi & 255) << 8 | (lo & 255)` over immutable
 * reads. is_immutable_value refuses that -- the update is an expression over
 * immutable leaves rather than one of them -- which drops the chain and leaves
 * the guard symbolic, so the loop never folds. */
#include <assert.h>

struct decoder
{
  int step;
  unsigned short cell[1000];
};

struct decoder VAR;

unsigned char nondet_uchar(void);

int main(void)
{
  for (VAR.step = 0; VAR.step < 1000; VAR.step = VAR.step + 1)
  {
    unsigned char hi = nondet_uchar();
    unsigned char lo = nondet_uchar();
    VAR.cell[VAR.step] = (unsigned short)(((hi & 0xff) << 8) | (lo & 0xff));
  }

  assert(VAR.step == 1000);
  return 0;
}
