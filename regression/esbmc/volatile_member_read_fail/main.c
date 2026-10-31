struct Flags
{
  volatile int changing;
  int stable;
};

struct Flags flags;

int main(void)
{
  int first = flags.changing;
  int second = flags.changing;
  __ESBMC_assert(first == second, "volatile field reads may differ");
}
