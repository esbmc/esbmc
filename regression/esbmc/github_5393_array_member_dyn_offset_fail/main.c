/* Counterpart of github_5393_array_member_dyn_offset: the struct read back from
 * the heap object carries the copied array, so asserting otherwise is
 * violated. */
#include <assert.h>
#include <stdlib.h>

struct s
{
  long a;
  long pair[2];
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
  tmpl.pair[0] = 6;
  tmpl.pair[1] = 7;
  struct s *p = (struct s *)(buf + i);
  *p = tmpl;
  struct s back = *p;
  assert(back.pair[1] != 7);
  free(buf);
  return 0;
}
