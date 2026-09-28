#include <cassert>
#include <string>

int main()
{
  std::string s(10, 'x');

  // Pins the reversed-parameter regression: with the arguments swapped the
  // model built a string of length (size_t)'x' == 120, so this assertion
  // would hold.  With the standard order it is length 10 and this fails.
  assert(s.length() == 120);

  return 0;
}
