// GitHub #6154: discriminating failing case for the refused float `++'.
// Falling back to normal unwinding must still evaluate the body, so the false
// claim f(k) == k + 99 is detected rather than passing vacuously.
int f(int n)
{
  double d = 0.0;
  for (int i = 0; i < 3; i++)
    d++;
  return n + (int)d;
}

int main()
{
  int k;
  __ESBMC_assert(__ESBMC_forall(&k, f(k) == k + 99), "should fail");
  return 0;
}
