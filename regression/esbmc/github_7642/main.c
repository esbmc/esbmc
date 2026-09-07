#include <assert.h>

unsigned char flag;

int main()
{
  flag = 0;
  assert(__atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == 0);
  assert(flag == 1);

  assert(__atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == 1);
  assert(flag == 1);

  __atomic_clear(&flag, __ATOMIC_SEQ_CST);
  assert(flag == 0);

  /* Any nonzero byte reads as already set, and the set value is still 1. */
  flag = 7;
  assert(__atomic_test_and_set(&flag, __ATOMIC_ACQUIRE) == 1);
  assert(flag == 1);

  return 0;
}
