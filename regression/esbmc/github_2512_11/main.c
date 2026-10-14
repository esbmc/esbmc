/* `g` is 2*offsetof(a, c) == 8, so `i` is &e.d and the write makes e.d 3 --
 * the assertion holds (clang, -fsanitize=address,undefined). It reported FAILED
 * only because the multiplied offsetof lost the object and the store went
 * nowhere; see docs/design/pointer-integer-provenance.md (#6545). */
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
