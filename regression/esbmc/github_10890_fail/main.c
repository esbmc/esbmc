/* #10890: the unaligned read of a packed struct inside a uint32_t array is
 * allowed, and the bytes it yields are the ones checked. */
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
  struct unaligned_u32 u = *p;
  assert(u.v == 0x11223344u);
  return 0;
}
