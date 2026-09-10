/* The write sits behind an `if`, so phi_function merges the object as
 * `VAR = if(g, then, else)` -- a shape constant_propagation carries at no arm,
 * so the counter beside the written member was dropped and the loop unwound
 * forever (InduByte/esbmc-evaluation#5, reported against the #7597 fix). The
 * merge is what decides, not the member's type, so a scalar member and an
 * array member both. No --unwind: the bound has to come from propagation. */
#include <assert.h>

struct plc
{
  int step;
  int out;
  int arr[6];
};

struct io
{
  int in;
};

struct plc VAR;
struct io IO;

int nondet_int(void);

int main(void)
{
  IO.in = nondet_int();
  /* A merge point, so IO carries no propagated constant afterwards, and a
   * bound that keeps the sums below from overflowing. */
  if (IO.in < -1000)
    IO.in = -1000;
  else if (IO.in > 1000)
    IO.in = 1000;

  for (VAR.step = 1; VAR.step <= 5; VAR.step = VAR.step + 1)
    if (IO.in > 0)
      VAR.out = IO.in + 1;
  assert(VAR.step == 6);
  /* The merged member still denotes what each branch left, not garbage:
   * VAR.out is a zero-initialised global the loop writes only when IO.in > 0. */
  assert(VAR.out == (IO.in > 0 ? IO.in + 1 : 0));

  for (VAR.step = 1; VAR.step <= 6; VAR.step = VAR.step + 1)
    if (IO.in > 0)
      VAR.arr[VAR.step] = IO.in + 2;
  assert(VAR.step == 7);
  assert(VAR.arr[3] == (IO.in > 0 ? IO.in + 2 : 0));

  return 0;
}
