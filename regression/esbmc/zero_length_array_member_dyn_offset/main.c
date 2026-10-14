/* zero_length_array_member_deref at a nondeterministic offset, where the struct
 * is rebuilt from the heap object's bytes by
 * construct_struct_ref_from_dyn_offs_rec: the copy writes the header and
 * leaves the byte after it untouched. */
#include <assert.h>
#include <stdlib.h>

struct s
{
  long a;
  long slots[0];
};

int main(void)
{
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 2);
  long *buf = calloc(4, sizeof(long));
  if (!buf)
    return 0;
  struct s tmpl;
  tmpl.a = 5;
  struct s *p = (struct s *)(buf + i);
  *p = tmpl;
  assert(buf[i] == 5);
  assert(buf[i + 1] == 0);
  free(buf);
  return 0;
}
