// The control for operator_new_nonliteral_size: the same allocation and the
// same in-bounds write, with the size a literal. This one verifies, which is
// what localises the defect to the size not being a literal.
#include <new>
#include <cassert>

int main()
{
  int *p = static_cast<int *>(::operator new(16));
  p[0] = 11;
  assert(p[0] == 11);
  ::operator delete(p);
  return 0;
}
