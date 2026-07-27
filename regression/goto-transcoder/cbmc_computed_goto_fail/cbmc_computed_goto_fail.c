/* Negative: the computed goto really selects the label, not a nondet branch. */
int main(void)
{
  void *const t[] = {&&a, &&b};
  int x = 0;
  goto *t[1];
a:
  x = 1;
  goto e;
b:
  x = 2;
  goto e;
e:
  __CPROVER_assert(x == 1, "wrong branch taken");
  return 0;
}
