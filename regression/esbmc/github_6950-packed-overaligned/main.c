#include <stdint.h>

/* An alignment specifier applies to the object, not to the type: clang reports
 * _Alignof(struct P) == 1 here yet places p at an 8-aligned address. ESBMC
 * stores both alignments in one "alignment" attribute, and alignment() takes
 * the minimum of it and the natural alignment when the type is packed
 * (src/util/expr/type_byte_size.cpp), so the object's alignment is capped at
 * the packed type's. Only specifiers up to the natural member maximum work. */

struct __attribute__((packed)) P
{
  char a;
  uint32_t b;
};

int main(void)
{
  _Alignas(8) struct P p;
  (void)p;
  __ESBMC_assert(((uintptr_t)&p % 8) == 0, "_Alignas(8) packed object is 8-aligned");
  return 0;
}
