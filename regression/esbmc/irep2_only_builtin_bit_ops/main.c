int main(void)
{
  __ESBMC_assert(__builtin_popcount(0xf0u) == 4, "popcount");
  __ESBMC_assert(__builtin_parity(0x7u) == 1, "parity of three set bits");
  __ESBMC_assert(__builtin_parity(0x3u) == 0, "parity of two set bits");
  __ESBMC_assert(__builtin_bswap16(0x1234u) == 0x3412u, "bswap16");
  __ESBMC_assert(__builtin_bswap32(0x12345678u) == 0x78563412u, "bswap32");
  __ESBMC_assert(
    __builtin_bswap64(0x0102030405060708ull) == 0x0807060504030201ull,
    "bswap64");
  return 0;
}
