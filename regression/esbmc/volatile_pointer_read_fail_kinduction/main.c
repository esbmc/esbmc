volatile int value;

int main(void)
{
  volatile int *pointer = &value;
  int first = *pointer;
  int second = *pointer;
  __ESBMC_assert(first == second, "volatile reads may differ");
}
