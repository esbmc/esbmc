/* The first test-and-set through a void pointer returns the old value 0. */
unsigned char flag;

int main()
{
  void *p = &flag;
  flag = 0;
  __ESBMC_assert(
    __atomic_test_and_set(p, __ATOMIC_SEQ_CST) == 1, "set returns old 1");
  return 0;
}
