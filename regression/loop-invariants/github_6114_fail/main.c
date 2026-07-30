// discussion #6114 (negative): the earlier quantifier invariant is not
// inductive (a[k] == 1 never holds, since the body writes 0). The fix must
// re-evaluate it after havoc, so its inductive step is caught (FAILED). Without
// the fix it keeps its vacuous pre-loop value and passes spuriously. No user
// assertion, so the only checkable property is the wrong invariant itself.
#define N 4
int main()
{
  int a[N];
  int i = 0;
  int k;
  __ESBMC_loop_invariant(
    __ESBMC_forall(&k, !(0 <= k) || !(k < i) || a[k] == 1));
  __ESBMC_loop_invariant(__ESBMC_forall(&k, k + 1 == 1 + k));
  __ESBMC_loop_invariant(0 <= i && i <= N);
  while (i < N)
  {
    a[i] = 0;
    ++i;
  }
}
