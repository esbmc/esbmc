/* __builtin_clzg is the width-generic count-leading-zeros: clzg(x, fb)
 * returns fb for a zero operand instead of being undefined. Unmodelled it
 * returned nondet, so anything built on it -- LLVM libc's cpp::countl_zero
 * prefers this spelling -- silently computed garbage (esbmc/esbmc#6925). */
#include <assert.h>
#include <stdint.h>

uint8_t nondet_u8(void);

int main(void)
{
  uint8_t v = nondet_u8();
  int g = __builtin_clzg(v, 8);
  assert(g >= 0 && g <= 8);

  assert(__builtin_clzg((uint8_t)0x80, 8) == 0);
  assert(__builtin_clzg((uint8_t)0x01, 8) == 7);
  assert(__builtin_clzg((uint8_t)0x00, 8) == 8);  /* the fallback argument */
  assert(__builtin_clzg((uint16_t)0x0100, 16) == 7);
  assert(__builtin_clzg((uint32_t)1u, 32) == 31);
  return 0;
}
