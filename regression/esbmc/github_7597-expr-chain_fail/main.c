/* The `with`-chain half of the same defect. do_simplify folds a `with` over a
 * propagated *literal* back into a literal, so a struct that still carries one
 * loses its counter on the aggregate-literal path instead. Merging VAR first
 * leaves it without a literal to fold into, so the writes stay `with` chains
 * and the chain path is the one that has to keep the counter. */
#include <assert.h>

struct plc
{
  int mode;
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
_Bool nondet_bool(void);

int main(void)
{
  IO.in = nondet_float();
  if (IO.in < -1.0e38f)
    IO.in = -1.0e38f;
  else if (IO.in > 1.0e38f)
    IO.in = 1.0e38f;

  /* VAR carries no literal past here, so `VAR.out = ..` stays a `with`. */
  if (nondet_bool())
    VAR.mode = 1;
  else
    VAR.mode = 2;

  for (VAR.step = 1; VAR.step <= 5; VAR.step = VAR.step + 1)
    VAR.out = IO.in + 1.0f;

  assert(VAR.step == 5);
  assert(VAR.mode == 1 || VAR.mode == 2);
  return 0;
}
