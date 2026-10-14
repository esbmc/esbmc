// The native "__ESBMC_uninterpreted_*" prefix (always modelled as an
// uninterpreted function, no flag) routes through the same handler as the
// "__CPROVER_uninterpreted_*" alias, so it shares the non-scalar-argument
// fix for GitHub #5369: a pointer argument must not abort the SMT backend.
unsigned long __ESBMC_uninterpreted_hasher(const void *key);

int main()
{
  int x = 7;
  const void *p = &x;
  unsigned long h = __ESBMC_uninterpreted_hasher(p);
  (void)h;
  __ESBMC_assert(x == 7, "unrelated property holds; native UF must not crash");
  return 0;
}
