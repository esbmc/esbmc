#include <stdint.h>

/* `#pragma pack(2)` puts b at offset 2, so a uint32_t access through a plain
 * pointer to it is misaligned -- clang traps this under
 * -fsanitize=alignment. A direct `p->b` read is not a violation: the compiler
 * knows the member's alignment and emits an unaligned load. */

#pragma pack(push, 2)
struct pack2_s
{
  char a;
  uint32_t b;
};
#pragma pack(pop)

int main(void)
{
  struct pack2_s obj;
  struct pack2_s *p = &obj;
  uint32_t direct = p->b;
  uint32_t *q = (uint32_t *)&(p->b);
  uint32_t z = *q;
  (void)direct;
  (void)z;
  return 0;
}
