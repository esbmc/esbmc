int __VERIFIER_nondet_int();
#define N 100000
int a[N];
int main()
{
  for (int i = 0; i < N; i++)
    a[i] = __VERIFIER_nondet_int();
  for (int x = 0; x < N - 1; x++)
    __ESBMC_assert(a[x] == a[x + 1], "distinct values can differ");
}
