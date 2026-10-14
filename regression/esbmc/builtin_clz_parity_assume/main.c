// Models for GCC/Clang builtins __builtin_clz, __builtin_parity and
// __builtin_assume (issue #4606). __builtin_clz(x) counts leading zero bits
// (undefined for x == 0); __builtin_parity(x) is the number of set bits mod 2;
// __builtin_assume(cond) states that cond holds.

int main()
{
  // __builtin_assume constrains a symbolic value.
  unsigned a;
  __builtin_assume(a == 0x80000000u);
  __ESBMC_assert(__builtin_clz(a) == 0, "clz of value with MSB set is 0");

  // clz on constants and on a symbolic value.
  __ESBMC_assert(__builtin_clz(1u) == 31, "clz(1) == 31");
  __ESBMC_assert(__builtin_clz(0xFFu) == 24, "clz(0xFF) == 24");
  __ESBMC_assert(
    __builtin_clzl(1ul) == (int)(sizeof(long) * 8 - 1), "clzl(1)");
  __ESBMC_assert(__builtin_clzll(1ull) == 63, "clzll(1) == 63");

  unsigned x;
  __builtin_assume(x == 16u);
  __ESBMC_assert(__builtin_clz(x) == 27, "clz(16) == 27");

  // parity = popcount mod 2.
  __ESBMC_assert(__builtin_parity(0x7u) == 1, "parity(0b111) == 1");
  __ESBMC_assert(__builtin_parity(0x6u) == 0, "parity(0b110) == 0");
  __ESBMC_assert(
    __builtin_parityll(0xFFFFFFFFFFFFFFFFull) == 0, "parity(64 ones) == 0");

  return 0;
}
