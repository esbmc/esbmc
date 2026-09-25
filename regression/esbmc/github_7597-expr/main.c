/* #7605 let a member write carry an immutable *read* -- `VAR.out = IO.in`.
 * A read wrapped in any operator is not one, so the object was still dropped
 * and the sibling counter with it, and the loop unwound forever (#7597 again,
 * reported against the merged fix). No --unwind here: the bound has to come
 * from propagation. One loop per operator shape that reaches symex as a
 * distinct node -- add, neg, mul, if, and a comparison widened to the member's
 * type. Each loop runs to its own bound so the five assertions differ: five
 * copies of one predicate would be folded to one by the assertion cache and
 * four of the shapes would go unchecked. */
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
    VAR.out = IO.in + 1.0f;
  assert(VAR.step == 6);

  for (VAR.step = 1; VAR.step <= 6; VAR.step = VAR.step + 1)
    VAR.out = -IO.in;
  assert(VAR.step == 7);

  for (VAR.step = 1; VAR.step <= 7; VAR.step = VAR.step + 1)
    VAR.out = IO.in * 2.0f;
  assert(VAR.step == 8);

  for (VAR.step = 1; VAR.step <= 8; VAR.step = VAR.step + 1)
    VAR.out = IO.in > 0.0f ? 1.0f : 2.0f;
  assert(VAR.step == 9);

  for (VAR.step = 1; VAR.step <= 9; VAR.step = VAR.step + 1)
    VAR.out = (float)(IO.in > 0.0f);
  assert(VAR.step == 10);

  return 0;
}
