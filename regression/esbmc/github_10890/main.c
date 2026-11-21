/* #10890: a packed struct has alignment 1, so reading it, or its member, from
 * inside an element of a uint32_t array is not misaligned. */
#include <assert.h>
#include <stdint.h>

struct __attribute__((packed)) unaligned_u32
{
  uint32_t v;
};

int main(void)
{
  uint32_t buf[2] = {0x11223344u, 0x55667788u};
  const struct unaligned_u32 *p =
    (const struct unaligned_u32 *)((const unsigned char *)buf + 1);
  uint32_t x = p->v;
  struct unaligned_u32 u = *p;
  assert(x == 0x88112233u);
  assert(u.v == 0x88112233u);
  return 0;
}
