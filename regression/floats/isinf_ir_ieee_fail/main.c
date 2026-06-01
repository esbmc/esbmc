// isinf must return false for values known to be within the finite float range.
// Claiming a value between 1.0 and 2.0 is infinite is unsound; ESBMC should
// find a counterexample (e.g. f = 1.5).
#include <assert.h>
#include <math.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float f = __VERIFIER_nondet_float();

  if(f >= 1.0f && f <= 2.0f)
    assert(isinf(f)); // wrong: f is finite here

  return 0;
}
