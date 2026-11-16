void __ESBMC_havoc_object(void *);
int a, b;
char c;
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

  // A string literal target is a constant and is left alone.
  char *s = __VERIFIER_nondet_int() ? &c : "lit";
  __ESBMC_havoc_object(s);
  if (s != &c)
    __ESBMC_assert(s[0] == 'l', "string literal untouched");
  return 0;
}
