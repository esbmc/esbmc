void __ESBMC_havoc_object(void *);
int a, b;
int main(void)
{
  a = 1;
  b = 2;
  int *p = __VERIFIER_nondet_int() ? &a : &b;
  __ESBMC_havoc_object(p);
  if (p == &a)
    __ESBMC_assert(b == 2, "b untouched when p points to a");
  else
    __ESBMC_assert(a == 1, "a untouched when p points to b");
  return 0;
}
