/* The write targets an array member, or a member of a nested struct, of the
 * same object that holds the counter. The refused update is then aggregate-
 * typed -- `VAR = with(VAR, "arr", with(VAR.arr, VAR.step, v))` -- and the
 * read it is pinned to, `VAR.arr`, is aggregate-typed too, so carrying it
 * needs is_immutable_value to admit a fixed-size aggregate read. Without that
 * the object is dropped, the counter never folds, and the loop unwinds forever
 * (InduByte/esbmc-evaluation#4, reported against the #7597 fix). No --unwind:
 * the bound has to come from propagation. */
#include <assert.h>

struct nested
{
  int out;
};

struct plc
{
  int step;
  int arr[6];
  struct nested in;
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
    VAR.arr[VAR.step] = IO.in + 1;
  assert(VAR.step == 5);
  /* The pinned read denotes the value written, not just any value. */
  assert(VAR.arr[3] == IO.in + 2);

  for (VAR.step = 1; VAR.step <= 6; VAR.step = VAR.step + 1)
    VAR.in.out = IO.in * 2;
  assert(VAR.step == 6);
  assert(VAR.in.out == IO.in * 3);

  return 0;
}
