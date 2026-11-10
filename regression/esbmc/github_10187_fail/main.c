// #10187: interval analysis gave every unnamed parameter of a prototype the
// same entry, so an int argument followed by a double one threw
// bad_variant_access.
#include <assert.h>

void h(int);
void k(double);

int main()
{
  int x = 2;
  h(x);
  k(1.0);
  assert(x == 3);
  return 0;
}
