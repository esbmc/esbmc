// Negative variant of cpp_clib_om (esbmc/esbmc#5298): proves the C++ clib OM
// body is actually verified (not silently skipped) by asserting a false
// property over its result.
extern "C" int __esbmc_probe_sum(int n);

int main()
{
  __ESBMC_assert(__esbmc_probe_sum(3) == 999, "intentionally false");
  return 0;
}
