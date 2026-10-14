/* Control for github_5393: the identical program writing the header field
 * directly instead of by struct assignment. It verifies, which is what
 * attributes that test's spurious violation to the assignment copying the
 * flexible array member rather than to the allocation or the tail read. */
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
  struct s *p = calloc(1, sizeof(struct s) + 2 * sizeof(struct e));
  if (!p)
    return 0;
  p->a = (void *)0x1234;
  assert(((unsigned char *)&p->slots[0])[1] == 0);
  return 0;
}
