// Regression for esbmc/esbmc#5298: a C++ operational model compiled into the
// embedded clib must be usable from a verified program. Before the fix, merely
// having a C++ source in the clib corrupted goto_convert ("non-code operand in
// this block") because the C++ frontend re-adjusted the C library functions.
extern "C" int __esbmc_probe_sum(int n);
extern "C" int __esbmc_probe_range();

int main()
{
  __ESBMC_assert(__esbmc_probe_sum(3) == 3, "0+1+2 == 3");
  __ESBMC_assert(__esbmc_probe_range() == 6, "1+2+3 == 6");
  return 0;
}
