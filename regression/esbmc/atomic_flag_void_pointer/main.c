/* __atomic_test_and_set and __atomic_clear act on the byte at a void
 * pointer. */
unsigned char flag;

int main()
{
  void *p = &flag;
  flag = 0;
  __ESBMC_assert(
    __atomic_test_and_set(p, __ATOMIC_SEQ_CST) == 0, "set returns old 0");
  __ESBMC_assert(flag == 1, "set stores 1");
  __ESBMC_assert(
    __atomic_test_and_set(p, __ATOMIC_SEQ_CST) == 1, "set returns old 1");
  __atomic_clear(p, __ATOMIC_SEQ_CST);
  __ESBMC_assert(flag == 0, "clear stores 0");
  return 0;
}
