/* Copying a struct with a zero-length array member (the GCC extension) into a
 * heap object rebuilds the struct from the object's bytes, member by member.
 * The member owns no bytes, so the copy writes the header and leaves the
 * calloc'd tail zero. Stitching the member together from zero bytes used to
 * stop ESBMC with "cannot read a zero-width object". */
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
  assert(((unsigned char *)&p->slots[0])[1] == 0);
  return 0;
}
