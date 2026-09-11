#include <stdint.h>

/* An object of unknown extent has no computable byte size, so the base
 * alignment falls back to the same stand-in convert_identifier_pointer() uses.
 * That stand-in has to keep the base reading as aligned, or the check would
 * report every access to such an object against a base it cannot constrain. */

__attribute__((annotate("__ESBMC_inf_size"))) uint64_t pool[1];

int main(void)
{
  uint64_t *p = (uint64_t *)((char *)pool + 1);
  uint64_t z = *p;
  (void)z;
  return 0;
}
