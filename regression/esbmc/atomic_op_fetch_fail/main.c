#include <assert.h>

int main()
{
  int i = 1;
  __sync_add_and_fetch(&i, 1);
  __atomic_add_fetch(&i, 1, __ATOMIC_SEQ_CST);
  assert(i == 1);
  return 0;
}
