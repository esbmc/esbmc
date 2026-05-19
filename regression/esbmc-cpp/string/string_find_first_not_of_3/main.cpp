// Edge cases for std::string::find_first_not_of:
//   - empty haystack with pos == 0 returns npos
//   - pos == length() returns npos
//   - all-matching haystack returns npos (regression for char-overload bug
//     where the return statement was outside the loop)
//   - basic mismatch search with explicit pos
#include <string>
#include <cassert>

int main()
{
  // empty haystack with pos == 0 returns npos
  std::string e("");
  assert(e.find_first_not_of("abc", 0) == std::string::npos);
  assert(e.find_first_not_of(std::string("abc"), 0) == std::string::npos);
  assert(e.find_first_not_of('a', 0) == std::string::npos);

  // pos == length() returns npos
  std::string s("hello");
  assert(s.find_first_not_of("xyz", 5) == std::string::npos);
  assert(s.find_first_not_of(std::string("xyz"), 5) == std::string::npos);
  assert(s.find_first_not_of('z', 5) == std::string::npos);

  // all characters match -> npos (char overload regression)
  std::string a("aaaa");
  assert(a.find_first_not_of('a') == std::string::npos);
  assert(a.find_first_not_of("a") == std::string::npos);
  assert(a.find_first_not_of(std::string("a")) == std::string::npos);

  // first non-match found from pos
  std::string t("aaab");
  assert(t.find_first_not_of('a') == 3);
  assert(t.find_first_not_of("a", 1) == 3);

  // empty C-string needle: every position is "not in" the empty set,
  // so the search returns pos (or npos if pos >= length()).
  assert(t.find_first_not_of("", 2) == 2);
  assert(t.find_first_not_of("", 0) == 0);

  return 0;
}
