// Copying a pointer's object representation byte by byte yields the same
// pointer (C11 6.2.6.1p4). Under --ir the byte operations cast the pointer to
// an integer, which the integer encoding cannot express.
#include <assert.h>
#include <string.h>
int g;
int main()
{
  int *q = &g, *r = 0;
  unsigned char b[sizeof q];
  memcpy(b, &q, sizeof q);
  memcpy(&r, b, sizeof q);
  assert(r == &g);
  return 0;
}
