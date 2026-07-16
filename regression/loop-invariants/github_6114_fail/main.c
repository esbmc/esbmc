// discussion #6114 (negative): a wrong earlier quantifier invariant must still
// be caught once it is re-evaluated after havoc, so the verdict stays FAILED.
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
    __ESBMC_assert(a[0] == 1, "a0 is one");
  }
}
