#include <assert.h>

int main()
{
  int i = 6;
  __atomic_fetch_nand(&i, 3, __ATOMIC_SEQ_CST);
  assert(i == 2);
  return 0;
}
