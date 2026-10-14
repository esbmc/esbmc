/* Only the interval bound 0 <= status <= 2 lets the inductive step prove
   this program. The assert(0) on the dead branch folds to ASSERT 0, so the
   bound must stay as tight when a failed assertion no longer ends its path
   (esbmc/esbmc#7900, from k-induction/github_1092_2_true). */
#include <assert.h>
int main()
{
  int status = 0;
  while (nondet_int())
  {
    if (!status)
      status = 1;
    else if (status == 1)
      status = 2;
    else if (status > 2)
      assert(0);
  }
  while (1)
    assert(status != 3);
}
