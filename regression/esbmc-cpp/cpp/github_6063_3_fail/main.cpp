// Negative direction for the round-3 STL gaps (github #6063): each new
// facility must carry a real value, not a nondet one, so a wrong expectation
// has to be reported as a violation.
#include <memory>
#include <cmath>
#include <string>
#include <cassert>

int main()
{
  int v = 42;
  int *p = std::addressof(v);
  *p = 3;
  assert(v == 42); // must fail: addressof aliases v, which is now 3

  assert(std::fpclassify(1.5) == FP_ZERO); // must fail: 1.5 is FP_NORMAL

  const char *text = "hello";
  std::string s(text, text + 5);
  assert(s.length() == 2); // must fail: the iterator pair spans 5 chars

  return 0;
}
