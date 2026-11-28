// In C++ the operand of __builtin_bit_cast is a glvalue, so a call's result
// arrives as a materialised temporary. Its bits, not its address, are cast.
typedef signed char s8x16 __attribute__((vector_size(16)));
typedef unsigned int u32x4 __attribute__((vector_size(16)));

static s8x16 make()
{
  s8x16 r = {1, 2, 3, 4, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -128};
  return r;
}

static long long make_ll()
{
  return -2;
}

int main()
{
  u32x4 u = __builtin_bit_cast(u32x4, make());
  __ESBMC_assert(u[0] == 0x04030201u, "lane 0");
  __ESBMC_assert(u[1] == 0xffffffffu, "lane 1");
  __ESBMC_assert(u[3] == 0x80000000u, "lane 3");
  __ESBMC_assert(
    __builtin_bit_cast(unsigned long long, make_ll()) == 0xfffffffffffffffeull,
    "scalar");
  return 0;
}
