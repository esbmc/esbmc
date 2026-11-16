void __ESBMC_havoc_object(void *);
int a, b;
int main(void)
{
  a = 1;
  b = 2;
  int *p = __VERIFIER_nondet_int() ? &a : &b;
  __ESBMC_havoc_object(p);
  if (p == &b)
    __ESBMC_assert(b == 2, "the object p points to is havocked");
  return 0;
}
