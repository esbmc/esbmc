// The SV-COMP shape: initialise, copy, then check through __VERIFIER_assert.
void __VERIFIER_assert(int cond) { __ESBMC_assert(cond, "cond"); }
#define N 100000
int a[N], b[N];
int main()
{
  for (int i = 0; i < N; i++)
    a[i] = 42;
  for (int i = 0; i < N; i++)
    b[i] = a[i];
  for (int x = 0; x < N; x++)
    __VERIFIER_assert(b[x] == 42);
}
