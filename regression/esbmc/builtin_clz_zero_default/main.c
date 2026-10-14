// Without --clz-zero-check, __builtin_clz(0) is not flagged as UB: clz is a
// pure value (width - popcount of the right-smeared argument), so clz(0) is the
// bit width. See #4606.
int main()
{
  __ESBMC_assert(__builtin_clz(0u) == 32, "clz(0) defaults to the bit width");
  return 0;
}
