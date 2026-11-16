struct Flags
{
  volatile int value;
};

struct Flags flag;

void check(int value)
{
  __ESBMC_assert(value == 0, "a volatile argument reads a fresh value");
}

int main(void)
{
  check(flag.value);
}
