struct Flags
{
  int value;
};

volatile struct Flags flag;

int main(void)
{
  __ESBMC_assert(flag.value == 0, "a volatile assertion reads a fresh value");
}
