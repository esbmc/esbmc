/* #7597: the loop counter is a member of the same struct whose other member
 * is written a symbolic value. Propagation worked at whole-object
 * granularity, so that one write dropped the counter with it, the guard never
 * folded and symex unwound forever. No --unwind here: the bound has to come
 * from propagation. */
#include <assert.h>

struct plc
{
  int step;
  float out;
};

struct io
{
  float in;
};

struct plc VAR;
struct io IO;

float nondet_float(void);

int main(void)
{
  IO.in = nondet_float();
  /* A merge point, so IO carries no propagated constant afterwards. */
  if (IO.in < -1.0e38f)
    IO.in = -1.0e38f;
  else if (IO.in > 1.0e38f)
    IO.in = 1.0e38f;

  for (VAR.step = 1; VAR.step <= 5; VAR.step = VAR.step + 1)
    VAR.out = IO.in;

  assert(VAR.step == 6);
  return 0;
}
