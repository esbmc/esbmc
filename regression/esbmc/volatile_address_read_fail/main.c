volatile int value;

int main(void)
{
  int first = *(&value);
  int second = *(&value);
  __ESBMC_assert(first == second, "volatile address reads may differ");
}
