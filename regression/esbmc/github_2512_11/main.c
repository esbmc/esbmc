#include <stddef.h>
#include <stdint.h>
void main() {
  struct a {
    int b;
    int c;
    int d;
  } e;
  uintptr_t f = (uintptr_t)&e;
  uintptr_t g = offsetof(struct a, c);
  g *= 2;
  int *i = (int *)(g + f);
  *i = 3;
  assert(e.d);
}
