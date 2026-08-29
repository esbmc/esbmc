#include <assert.h>
#include <stdint.h>

int main()
{
  int x = 7;
  int *p = (int *)(uintptr_t)&x;
  assert(*p == 7);
  return 0;
}
