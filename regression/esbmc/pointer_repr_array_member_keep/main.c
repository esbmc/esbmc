#include <assert.h>

struct S
{
  int *a[2];
  char buf[8];
};

unsigned nondet_uint();

// Pointers held in an array member are not rebuilt as themselves, but a byte
// written into buf still leaves them unchanged.
int main()
{
  int k, j;
  struct S s;
  s.a[0] = &k;
  s.a[1] = &j;
  unsigned char *bytes = (unsigned char *)&s;
  bytes[2 * sizeof(int *) + nondet_uint() % 8] = 'a';
  assert(s.a[0] == &k);
}
