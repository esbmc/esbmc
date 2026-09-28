/* GNU labels-as-values / computed goto. CBMC lowers `goto *p` into a concrete
   IF-chain comparing label addresses (address_of(label)); ESBMC has no
   label-address node, so the adapter maps each label to a unique (void *)K
   constant so those equality comparisons reproduce CBMC's control flow. */
int dispatch(int sel)
{
  static void *const table[] = {&&zero, &&one, &&two};
  int acc = 0;
  goto *table[sel];
zero:
  acc = 10;
  goto done;
one:
  acc = 11;
  goto done;
two:
  acc = 12;
  goto done;
done:
  return acc;
}

int main(void)
{
  __CPROVER_assert(dispatch(0) == 10, "select 0");
  __CPROVER_assert(dispatch(1) == 11, "select 1");
  __CPROVER_assert(dispatch(2) == 12, "select 2");

  int k;
  __CPROVER_assume(k >= 0 && k <= 2);
  int r = dispatch(k);
  __CPROVER_assert(r == 10 || r == 11 || r == 12, "symbolic select");
  return 0;
}
