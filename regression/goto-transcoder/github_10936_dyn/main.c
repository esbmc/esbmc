/* Built with the dereference of the cast below rewritten in CBMC's symbol table
 * to byte_extract_little_endian(src, 0) of the target struct type, the node
 * Kani emits for a transmute such as NonNull::<[u8]>::from_ref. */
#include <assert.h>
struct vtable { unsigned long size, align; };
struct fat { int *data; const struct vtable *vt; };
struct pattern { struct fat f; };
struct nonnull { struct pattern pointer; };
const struct vtable vt_int = {4, 4};
int x;
int main(void)
{
  struct fat p = {&x, &vt_int};
  /* mkbe.py rewrites this into byte_extract_little_endian(p, 0), as Kani's
   * NonNull::<dyn T>::new_unchecked does */
  struct nonnull nn = *(struct nonnull *)&p;
  assert(nn.pointer.f.vt->align == 4);
  return 0;
}
