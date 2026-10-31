struct NvBool
{
  int value;
};

volatile struct NvBool flag;

int main(void)
{
  int first = flag.value;
  int second = flag.value;
  __ESBMC_assert(first == second, "volatile struct member reads may differ");
}
