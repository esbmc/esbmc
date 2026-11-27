/* Built with each dereference of a cast below rewritten in CBMC's symbol table
 * to byte_extract_little_endian(src, 0) of the target type, the node Kani
 * emits for a transmute such as NonNull::dangling() reading an Alignment. */
#include <assert.h>
struct repr { unsigned long tag; };
struct inner { unsigned long v; };
struct wrap { struct inner i; };
struct nonnull { int *p; };
unsigned long nondet_ulong(void);
int nondet_int(void);

int main(void)
{
  unsigned long t = nondet_ulong();
  struct repr a = {t};
  struct wrap b = *(struct wrap *)&a;
  assert(b.i.v != t);

  int x = nondet_int();
  struct nonnull s = {&x};
  int *q = *(int **)&s;
  assert(q == &x);

  char *c = *(char **)&q;
  assert(c == (char *)&x);
  return 0;
}
