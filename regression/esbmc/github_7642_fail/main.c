#include <assert.h>

unsigned char flag;

int main()
{
  flag = 0;
  /* Returns the OLD contents, so this is 0, not 1. */
  assert(__atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == 1);
  return 0;
}
