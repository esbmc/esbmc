/* The byte at the cap must still allocate: PTRDIFF_MAX is the largest request
   that succeeds, not the first that fails. --force-malloc-success removes the
   ordinary may-fail outcome, leaving only the cap. Kept to a single allocation
   -- pairing it with the above-cap case exhausts the address space, and the
   layout constraints then go UNSAT and prove both halves vacuously. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
  char *at = malloc((size_t)PTRDIFF_MAX);
  assert(at != NULL);
  return 0;
}
