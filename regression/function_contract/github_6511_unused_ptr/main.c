// The contract states no extent for `out`, but nothing here reads or writes
// through it, so the unstated-extent advice does not apply (#6511).
void f(int *out, int v)
{
  __ESBMC_requires(v > 0);
  __ESBMC_ensures(1);
  (void)out;
}

int main(void)
{
  return 0;
}
