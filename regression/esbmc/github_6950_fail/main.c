#include <stdint.h>

/* The declaration's specifier is carried across type-symbol substitution, so
 * the 16-byte claim must hold -- an implementation that took the minimum of
 * the two alignments, or dropped the specifier, would violate it here. The
 * 32-byte claim over-states the specifier and must still fail. */

struct N
{
  char a;
  uint32_t b;
};

int main(void)
{
  _Alignas(16) struct N s;
  (void)s;
  __ESBMC_assert(((uintptr_t)&s % 16) == 0, "_Alignas(16) object is 16-aligned");
  __ESBMC_assert(((uintptr_t)&s % 32) == 0, "over-claimed 32-byte alignment");
  return 0;
}
