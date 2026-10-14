// #7897: an operand that is not a symbol is bound to a temporary, so the
// dereference is still checked instead of being replicated per lane.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4i a = {1, 2, 3, 4};
  v4i *p = (v4i *)0;
  v4i m = *p == a;
  assert(m[0] == -1);
  return 0;
}
