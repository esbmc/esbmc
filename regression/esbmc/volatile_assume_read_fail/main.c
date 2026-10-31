struct Flags
{
  int value;
};

volatile struct Flags flag;

int main(void)
{
  __ESBMC_assume(flag.value != 0);
  __ESBMC_assert(0, "a volatile assumption can keep this path reachable");
}
