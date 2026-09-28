/* #7597's shape at the propagation cap: the counter is a member of the struct
 * whose array member takes one symbolic write per iteration, so the guard
 * folds only while those writes stay within symbolic_chain_bound. Anything
 * above 128 hung before the cap was raised; 1000 leaves margin under it, so a
 * future counted update turns this into a failure rather than a hang. */
#include <assert.h>

struct plc
{
  int step;
  float out[1000];
};

struct plc VAR;

float nondet_float(void);

int main(void)
{
  for (VAR.step = 0; VAR.step < 1000; VAR.step = VAR.step + 1)
    VAR.out[VAR.step] = nondet_float();

  assert(VAR.step == 1000);
  return 0;
}
