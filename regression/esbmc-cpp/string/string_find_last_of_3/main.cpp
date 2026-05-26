// Edge cases for std::string::find_last_of:
//   - default pos == npos must not overflow the precondition
//   - position 0 must be reachable
//   - pos beyond length() clamps to length()-1
//   - empty haystack / needle returns npos
#include <string>
#include <cassert>

int main()
{
  // position 0 must be reachable
  std::string s1("/abc");
  assert(s1.find_last_of("/") == 0);

  // not found returns npos
  std::string s2("abc");
  assert(s2.find_last_of("xyz") == std::string::npos);

  // explicit pos clamps the search window
  std::string s3("aXbX");
  assert(s3.find_last_of("X", 2) == 1);

  // pos beyond length() is clamped, not an overflow
  std::string s4("ab");
  assert(s4.find_last_of("a", 100) == 0);

  // single-char overload with large pos
  std::string s5("xxa.b");
  assert(s5.find_last_of('.', 100) == 3);

  // empty haystack returns npos
  std::string s6("");
  assert(s6.find_last_of("a") == std::string::npos);

  return 0;
}
