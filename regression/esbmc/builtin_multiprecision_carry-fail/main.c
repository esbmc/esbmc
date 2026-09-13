#include <assert.h>

/* Negative half of builtin_multiprecision_carry: the carry-out of the second
 * partial addition must be reported, not dropped. */
int main(void)
{
  unsigned c;
  assert(__builtin_addc(0xffffffffu, 0u, 1u, &c) == 0u && c == 0u);
  return 0;
}
