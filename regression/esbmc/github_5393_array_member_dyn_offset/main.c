/* github_5393_array_member at a nondeterministic offset, where the struct is
 * rebuilt from the heap object's bytes by construct_struct_ref_from_dyn_offs_rec. */
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
  assert(buf[i + 1] == 6 && buf[i + 2] == 7);
  struct s back = *p;
  assert(back.pair[0] == 6 && back.pair[1] == 7);
  free(buf);
  return 0;
}
