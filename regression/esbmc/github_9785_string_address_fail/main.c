/* #9785: an address taken of something other than a symbol, here a string
 * literal, has no start symbol of its own, so its integer value is read from
 * the address space. */
#include <stdint.h>
int main(void)
{
  uintptr_t a = (uintptr_t)"abc";
  __ESBMC_assert(a == 0, "a string literal's address");
  return 0;
}
