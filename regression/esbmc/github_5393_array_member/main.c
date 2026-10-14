/* #5393 control for the zero-width checks: an array member that does own bytes
 * is still copied into, and read back from, a heap object. */
#include <assert.h>
#include <stdlib.h>

struct s
{
  void *a;
  long pair[2];
};

int main(void)
{
  struct s tmpl;
  tmpl.a = (void *)0x1234;
  tmpl.pair[0] = 6;
  tmpl.pair[1] = 7;
  struct s *p = calloc(1, sizeof(struct s) + sizeof(long));
  if (!p)
    return 0;
  *p = tmpl;
  assert(p->pair[0] == 6 && p->pair[1] == 7);
  struct s back = *p;
  assert(back.pair[0] == 6 && back.pair[1] == 7);
  return 0;
}
