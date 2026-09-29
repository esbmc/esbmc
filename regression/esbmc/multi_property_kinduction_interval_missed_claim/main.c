/* esbmc/esbmc#7900 D3 (program from #1361): assert(2==3) folds to ASSERT 0,
   and the post-k-induction interval pass used to treat it as the end of the
   path. The loop-head bound it inserted (i == 0) then made assert(1==2)
   unreachable in the inductive step once the first claim was cleared. */
#include <assert.h>
int main()
{
  for (int i = 0; i < 2; i++)
  {
    assert(2 == 3);
  }
  assert(1 == 2);
  return 0;
}
