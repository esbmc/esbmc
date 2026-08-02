// Regression for esbmc/esbmc#5298: with a C++ operational model present in the
// embedded clib, lowering a C for-loop must still succeed. The bug made
// goto_convert abort with "non-code operand in this block" because the C++
// frontend re-adjusted the C library functions, and clang_c_adjust::adjust_for
// was not idempotent (it re-wrapped the for-loop, leaving a stray nil operand).
int main()
{
  int s = 0;
  for (int i = 0; i < 3; i++)
    s += i;
  __ESBMC_assert(s == 3, "0+1+2 == 3");
  return 0;
}
