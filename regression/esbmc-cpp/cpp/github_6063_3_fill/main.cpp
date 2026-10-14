// The basic_string iterator-pair constructor added for github #6063 must not
// swallow the fill constructor: with both arguments integral, an unconstrained
// template deduces InputIt=int and wins on an exact match, so `string(3, 42)`
// would treat 3 and 42 as iterators. libstdc++ constrains its iterator-pair
// constructor the same way (_RequireInputIter).
#include <string>
#include <cassert>

int main()
{
  std::string fill(4, 'x');
  assert(fill.length() == 4);
  assert(fill[0] == 'x');
  assert(fill[3] == 'x');

  std::string fill2(3, 42);
  assert(fill2.length() == 3);
  assert(fill2[0] == 42);

  // The iterator-pair form itself still resolves for a raw pointer pair.
  const char *text = "hello";
  std::string s(text, text + 5);
  assert(s.length() == 5);

  return 0;
}
