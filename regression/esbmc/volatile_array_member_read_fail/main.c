struct Flags
{
  int values[2];
};

volatile struct Flags flags;

int main(void)
{
  int first = flags.values[1];
  int second = flags.values[1];
  __ESBMC_assert(first == second, "volatile array member reads may differ");
}
