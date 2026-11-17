// A store on the summarised array keeps it a lambda: without that the solver
// needs a forall over the whole array and cannot build a counterexample.
extern unsigned __VERIFIER_nondet_uint(void);
#define N 100000
int a[N];
int main()
{
  for (int i = 0; i < N; i++)
    a[i] = i;
  a[3] = 7;
  unsigned j = __VERIFIER_nondet_uint() % N;
  __ESBMC_assert(j == 3 || a[j] == (int)j, "init");
}
