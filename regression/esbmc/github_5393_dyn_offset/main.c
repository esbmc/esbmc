/* #5393 at a nondeterministic offset: the struct is rebuilt from the heap
 * object's bytes by construct_struct_ref_from_dyn_offs_rec rather than the
 * constant-offset path github_5393 takes. Assigning the struct must write only
 * its header and leave the calloc'd zero that follows it in place. */
#include <assert.h>
#include <stdlib.h>

struct s
{
  long a;
  long slots[];
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
  struct s back = *p;
  assert(back.a == 5);
  free(buf);
  return 0;
}
