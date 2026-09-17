/* Counterpart of zero_length_array_member_deref: the copy leaves the calloc'd
 * tail zero, so asserting that it set the tail is violated. */
#include <assert.h>
#include <stdlib.h>

struct e
{
  void *k, *v;
};
struct s
{
  void *a;
  struct e slots[0];
};

int main(void)
{
  struct s tmpl;
  tmpl.a = (void *)0x1234;
  struct s *p = calloc(1, sizeof(struct s) + 2 * sizeof(struct e));
  if (!p)
    return 0;
  *p = tmpl;
  assert(p->a == (void *)0x1234);
  assert(p->slots[0].k != 0);
  return 0;
}
