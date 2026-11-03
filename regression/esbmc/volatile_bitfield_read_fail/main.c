struct Flags
{
  volatile unsigned int small : 3;
};

struct Flags flags;

int main(void)
{
  unsigned int first = flags.small;
  unsigned int second = flags.small;
  __ESBMC_assert(first == second, "volatile bitfield reads may differ");
}
