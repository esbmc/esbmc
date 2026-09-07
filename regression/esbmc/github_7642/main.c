/* __ESBMC_assert rather than <assert.h>: MSVC expands assert(e) to the
 * short-circuit form (!!(e)) || (_wassert(...), 0), whose call is dropped
 * when e is resolvable at conversion time, so a passing assert leaves no
 * claim to pin on Windows. */

unsigned char flag;

int main()
{
  flag = 0;
  __ESBMC_assert(
    __atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == 0, "set returns old 0");
  __ESBMC_assert(flag == 1, "set stores 1");

  __ESBMC_assert(
    __atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == 1, "set returns old 1");
  __ESBMC_assert(flag == 1, "set leaves 1");

  __atomic_clear(&flag, __ATOMIC_SEQ_CST);
  __ESBMC_assert(flag == 0, "clear stores 0");

  /* Any nonzero byte reads as already set, and the set value is still 1. */
  flag = 7;
  __ESBMC_assert(
    __atomic_test_and_set(&flag, __ATOMIC_ACQUIRE) == 1, "nonzero reads as set");
  __ESBMC_assert(flag == 1, "set normalises to 1");

  return 0;
}
