#include <string>
#include <iterator>
#include <cassert>

int main()
{
  // Naming the traits typedef is the point; the assertion is deliberately
  // false so the property is refuted rather than vacuous.
  std::iterator_traits<std::string::iterator>::difference_type d = 3;
  assert(d == 4);
  return 0;
}
