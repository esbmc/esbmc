/* #7597, the shape where the struct carries no literal at all: the branch
 * above the loop leaves VAR as a phi symbol, so the counter folds only if a
 * `with` chain over that symbol may carry the symbolic sibling write. */
#include <assert.h>

struct plc
{
  int step;
  int out;
};

struct plc VAR;

int nondet_int(void);

int main(void)
{
  int in = nondet_int();

  if (nondet_int())
    VAR.out = 1;
  else
    VAR.out = 2;

  for (VAR.step = 1; VAR.step <= 5; VAR.step = VAR.step + 1)
    VAR.out = in;

  assert(VAR.step == 6);
  assert(VAR.out == in);
  return 0;
}
