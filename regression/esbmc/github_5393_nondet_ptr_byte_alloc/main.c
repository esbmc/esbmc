/* A pointer with no value-set provenance (a nondet one) does not survive a
   round trip through an untyped byte allocation: the store narrows to a
   single byte, so the assumed non-nullness is lost on the read back.

   This is the root cause of the SV-COMP false alarms in #5393 / #5394 --
   aws_hash_table_init copies its nondet hash_fn/equals_fn into a calloc'd
   hash_table_state, and calloc's model allocates through malloc(total_size)
   with a runtime size, which leaves the object untyped.

   The three sibling tests pin the boundary: concrete pointers, integers and
   typed allocations all survive the same shape. */
#include <stdlib.h>

void *nondet_voidp(void);

int main(void)
{
  void *q = nondet_voidp();
  __ESBMC_assume(q != 0);

  char *raw = malloc(64);
  if (raw == 0)
    return 0;

  void **s = (void **)raw;
  *s = q;

  __ESBMC_assert(*s != 0, "a stored non-null pointer reads back non-null");
  return 0;
}
