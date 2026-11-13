// Each element gets its own nondet value.
int __VERIFIER_nondet_int();
#define N 100000
int a[N], b[N];
int main()
{
  for (int i = 0; i < N; i++)
    a[i] = __VERIFIER_nondet_int();
  for (int i = 0; i < N; i++)
    b[i] = a[i];
  for (int x = 0; x < N; x++)
    __ESBMC_assert(b[x] == a[x], "copied");
}
