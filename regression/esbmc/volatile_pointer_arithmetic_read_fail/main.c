volatile int values[2];

int main(void)
{
  volatile int *pointer = &values[0];
  int first = *(pointer + 1);
  int second = *(pointer - (-1));
  __ESBMC_assert(first == second, "volatile pointer arithmetic reads may differ");
}
