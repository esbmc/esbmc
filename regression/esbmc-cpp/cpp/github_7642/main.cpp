#include <cassert>

unsigned char flag;

int main()
{
  flag = 0;
  assert(__atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == false);
  assert(flag == 1);

  assert(__atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) == true);

  __atomic_clear(&flag, __ATOMIC_SEQ_CST);
  assert(flag == 0);

  return 0;
}
