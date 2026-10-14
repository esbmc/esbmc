/* Counterpart of github_5393_array_member: the struct read back from the heap
 * object carries the copied array, so asserting otherwise is violated. */
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
  struct s back = *p;
  assert(back.pair[1] != 7);
  return 0;
}
