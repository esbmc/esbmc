#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S { int x; int y; int z; } s = {.x = 1};
  uintptr_t u = (uintptr_t)&s;
  u += 4;
  u -= 4;
  int *p = (int *)u;
  *p = 3;
  assert(s.x == 3);
  return 0;
}
