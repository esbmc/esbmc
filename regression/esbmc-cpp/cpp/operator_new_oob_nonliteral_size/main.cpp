// The negative half of operator_new_nonliteral_size: allocating n bytes must
// not make the bounds check vacuous, so an access past the requested size is
// still a violation.
#include <new>
#include <cassert>

int main()
{
  unsigned long n = 8;
  char *p = static_cast<char *>(::operator new(n));
  p[8] = 1; // one past the end
  assert(p[8] == 1);
  ::operator delete(p);
  return 0;
}
