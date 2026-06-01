// Edge cases for std::string::rfind (const char *):
//   - pos > length() clamps to length() (per the standard)
//   - pos == length() finds the last occurrence at the end
//   - empty needle returns min(pos, length())
//   - needle longer than haystack returns npos
//   - empty haystack returns npos
#include <string>
#include <cassert>

int main()
{
  std::string s("abcabc");

  // pos > length() clamps and finds the last occurrence
  assert(s.rfind("b", 100) == 4);
  assert(s.rfind("bc", 100, 2) == 4);

  // pos == length() finds the last occurrence at the end
  assert(s.rfind("c", 6) == 5);

  // empty needle returns min(pos, length())
  assert(s.rfind("", 3, 0) == 3);
  assert(s.rfind("", 100, 0) == 6);

  // needle longer than haystack returns npos
  assert(s.rfind("abcabcd") == std::string::npos);
  assert(s.rfind("abcabcd", 100, 7) == std::string::npos);

  // empty haystack returns npos
  std::string e("");
  assert(e.rfind("a") == std::string::npos);
  assert(e.rfind("a", 0, 1) == std::string::npos);

  // explicit pos limits the search window
  assert(s.rfind("b", 2) == 1);

  // pos == length() with multi-char needle exercises the max_start clamp
  assert(s.rfind("abc", 6) == 3);

  return 0;
}
