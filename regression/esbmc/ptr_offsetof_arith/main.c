#include <stdint.h>
#include <assert.h>
#include <stddef.h>

int main(void)
{
  struct S { int x; int y; int z; } s = {.y = 42};
  uintptr_t v = (uintptr_t)&s.x;
  uintptr_t u = offsetof(struct S, y);
  int *p = (int *)(u + v);
  *p = 3;
  assert(s.y == 3);
  return 0;
}
