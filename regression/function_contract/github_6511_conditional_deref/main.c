// `out` is dereferenced on one path only. Suppressing the unstated-extent
// advice here would be wrong: this is exactly the contract whose bounds check
// fails once v > 100 is reachable (#6511).
void f(int *out, int v)
{
  __ESBMC_requires(v > 0);
  __ESBMC_ensures(1);
  if (v > 100)
    *out = v;
}

int main(void)
{
  return 0;
}
