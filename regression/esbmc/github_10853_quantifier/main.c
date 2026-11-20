int a[5];
int main(void)
{
  int i = 0;
  int k;
  __ESBMC_loop_invariant(0 <= i && i <= 5 && __ESBMC_forall(&k, !(0 <= k && k < i) || a[k] == 0));
  for (; i < 5; i++)
    a[i] = 0;
  return 0;
}
