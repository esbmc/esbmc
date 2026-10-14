// A "__CPROVER_uninterpreted_*" function whose argument is a pointer (here a
// "const void *", as in CBMC's aws-c-common hash-table harnesses). A pointer
// lowers to a tuple sort, which the SMT backend cannot use as an uninterpreted-
// function domain; modelling it as a native uninterpreted function used to
// abort ESBMC in to_solver_smt_sort (GitHub #5369). The signature is now
// detected as non-scalar and the result is modelled as a fresh nondet value
// (sound over-approximation), so encoding completes and an unrelated property
// is proved without a crash.
unsigned long __CPROVER_uninterpreted_hasher(const void *key);

int main()
{
  int x = 3;
  const void *p = &x;
  unsigned long h = __CPROVER_uninterpreted_hasher(p);
  (void)h;
  __ESBMC_assert(x == 3, "unrelated property holds; UF call must not crash");
  return 0;
}
