/* Negative counterpart of github_7597-chain-dropped: the local counter still
 * folds to 1025 after the chain over `a` is dropped, so the claim below is
 * refuted rather than the loop unwinding forever. */
#include <assert.h>

int a[1025];

int nondet_int(void);

int main(void)
{
  int i;

  for (i = 0; i < 1025; i = i + 1)
    a[i] = nondet_int();

  assert(i == 1024);
  return 0;
}
