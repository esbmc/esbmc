// Pins __builtin_clz to its real value: clz(1) is 31 on a 32-bit int, so the
// (wrong) expectation of 30 must fail. Guards against an off-by-one or nondet
// model. See #4606.
int main()
{
  __ESBMC_assert(__builtin_clz(1u) == 30, "clz(1) is 31, not 30");
  return 0;
}
