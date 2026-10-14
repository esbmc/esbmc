/* Negative counterpart of github_7597-chain-bound: the counter folds to 1000,
 * so the claim below is refuted. It is reached only once the loop terminates,
 * which needs the whole 1000-update chain to be carried. */
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

  assert(VAR.step == 999);
  return 0;
}
