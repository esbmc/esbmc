#include <cassert>

unsigned char flag;

int main()
{
  flag = 1;
  __atomic_clear(&flag, __ATOMIC_RELEASE);
  /* clear() stores 0, so the byte is not 1 any more. */
  assert(flag == 1);
  return 0;
}
