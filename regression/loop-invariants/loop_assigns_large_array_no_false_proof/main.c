/* Past the element-wise cap the loop rule used to fall back to assuming the
 * array wholly unchanged -- while the clause grants writing global[3]. A
 * hypothesis stronger than the truth proves things that are false, and this
 * assertion, which the loop plainly falsifies, was proved.
 *
 * The constraint is now withheld rather than weakened into a falsehood: the
 * loop simply gets no frame hypothesis for this array, so the assertion is
 * refuted as it should be. Bitwuzla used to abort here on equality over
 * constant arrays; with no array equality emitted, both solvers now answer. */
int global[512];

int main()
{
  int i = 0;

  __ESBMC_loop_assigns(i, global[3]);
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  while (i < 4)
  {
    global[3] = 1;
    i++;
  }

  __ESBMC_assert(global[3] == 0, "FALSE: the loop wrote global[3] = 1");
  return 0;
}
