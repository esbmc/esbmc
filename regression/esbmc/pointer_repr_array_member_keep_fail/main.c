#include <assert.h>

struct S
{
  int *a[2];
  char buf[8];
};

unsigned nondet_uint();

// Pointers held in an array member are not rebuilt as themselves. The byte
// written may land in a[0], which then changes.
int main()
{
  int k, j;
  struct S s;
  s.a[0] = &k;
  s.a[1] = &j;
  unsigned char *bytes = (unsigned char *)&s;
  bytes[nondet_uint() % sizeof(s)] = 0xff;
  assert(s.a[0] == &k);
}
