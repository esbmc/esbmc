/* Built with the dereference of the cast below rewritten in CBMC's symbol table
 * to byte_extract_little_endian(src, 0) of the target struct type, the node
 * Kani emits for a transmute such as NonNull::<[u8]>::from_ref. */
#include <assert.h>
struct slice { unsigned char *data; unsigned long len; };
struct inner { struct slice s; };
struct outer { struct inner p; };
int nondet_int(void);

int main(void)
{
  int n = nondet_int();
  struct slice sl = {(unsigned char *)&n, sizeof n};
  struct outer o = *(struct outer *)&sl;
  assert(*o.p.s.data == (unsigned char)n);
  return 0;
}
