struct Flags
{
  volatile unsigned int small : 3;
  volatile signed int signed_small : 3;
};

struct Flags flags;

int main(void)
{
  unsigned int small = flags.small;
  int signed_small = flags.signed_small;
  __ESBMC_assert(small <= 7, "an unsigned 3-bit read stays in range");
  __ESBMC_assert(
    signed_small >= -4 && signed_small <= 3,
    "a signed 3-bit read stays in range");
}
