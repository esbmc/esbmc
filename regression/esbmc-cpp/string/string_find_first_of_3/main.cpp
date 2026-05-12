// Edge cases for std::string::find_first_of:
//   - empty haystack with pos == 0 returns npos
//   - pos == length() returns npos (all overloads)
//   - pos > length() returns npos
//   - pos == 0 finds index 0 when match is at the start
#include <string>
#include <cassert>

int main()
{
  // empty haystack with pos == 0 returns npos
  std::string e("");
  assert(e.find_first_of("abc", 0) == std::string::npos);
  assert(e.find_first_of(std::string("abc"), 0) == std::string::npos);
  assert(e.find_first_of('a', 0) == std::string::npos);

  // pos == length() returns npos (the bug fix)
  std::string s("hello");
  assert(s.find_first_of("h", 5) == std::string::npos);
  assert(s.find_first_of(std::string("h"), 5) == std::string::npos);
  assert(s.find_first_of('h', 5) == std::string::npos);

  // pos > length() returns npos
  assert(s.find_first_of("h", 100) == std::string::npos);

  // default pos finds index 0 when match at start
  assert(s.find_first_of("h") == 0);
  assert(s.find_first_of('h') == 0);

  // mid-string match with explicit pos
  std::string t("abcdef");
  assert(t.find_first_of("xc", 1) == 2);

  // empty C-string needle returns npos
  assert(t.find_first_of("", 0) == std::string::npos);
  assert(t.find_first_of("", 2) == std::string::npos);

  return 0;
}
