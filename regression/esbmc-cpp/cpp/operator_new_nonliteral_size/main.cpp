// ::operator new(n) with a non-literal n must allocate n bytes. It used to
// model a one-byte object, so even the first in-bounds write was reported out
// of bounds; operator_new_oob_nonliteral_size is the negative half, checking a
// genuine overrun is still caught.
#include <new>
#include <cassert>

int main()
{
  unsigned long n = 16;
  int *p = static_cast<int *>(::operator new(n));
  p[0] = 11;
  assert(p[0] == 11);
  ::operator delete(p);
  return 0;
}
