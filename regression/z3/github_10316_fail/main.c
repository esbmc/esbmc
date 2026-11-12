// #10316: Z3 aborted on ROUND_TO_AWAY under a symbolic rounding mode.
extern int __ESBMC_rounding_mode;
int nondet_int(void);

int main()
{
  __ESBMC_rounding_mode = nondet_int();
  // 1 + 2^-24 is a tie: ties-away gives 1 + 2^-23, ties-even gives 1.
  float x = 1.0f + 0x1p-24f;
  if (__ESBMC_rounding_mode == 1)
    __ESBMC_assert(x == 1.0f, "ties round to even");
  return 0;
}
