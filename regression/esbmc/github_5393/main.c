/* KNOWNBUG (#5393). Assigning a struct whose type has a flexible array member
 * must copy only the header: C17 6.7.2.1p18 says the size of the structure is
 * as if the flexible array member were omitted. ESBMC copies the flexible tail
 * as well, so the assignment overwrites the calloc'd zeroes that follow the
 * header with the source local's indeterminate bytes, and a read of the tail
 * reports a spurious violation.
 *
 * This is the SV-COMP aws_hash_table_init_bounded_harness false alarm reduced:
 * there the tail is the hash table's slot array, zeroed by aws_mem_calloc and
 * then apparently clobbered by `*state = *template`, so assert_all_zeroes over
 * the slots fails on a correct program. See github_5393_control for the same
 * program without the struct assignment, which verifies. */
#include <assert.h>
#include <stdlib.h>

struct e
{
  void *k, *v;
};
struct s
{
  void *a;
  struct e slots[];
};

int main(void)
{
  struct s tmpl;
  tmpl.a = (void *)0x1234;
  struct s *p = calloc(1, sizeof(struct s) + 2 * sizeof(struct e));
  if (!p)
    return 0;
  *p = tmpl;
  assert(((unsigned char *)&p->slots[0])[1] == 0);
  return 0;
}
