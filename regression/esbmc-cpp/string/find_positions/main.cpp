#include <cassert>
#include <string>

int main()
{
  std::string s = "abcabc";

  assert(s.find("abc") == 0);
  assert(s.find("abc", 1) == 3);
  assert(s.find("bc") == 1);
  assert(s.find("zz") == std::string::npos);
  assert(s.find('c') == 2);
  assert(s.find('c', 3) == 5);
  assert(s.find('z') == std::string::npos);

  // An empty needle matches at pos whenever pos <= size().
  assert(s.find("") == 0);
  assert(s.find("", 2) == 2);
  assert(s.find("", s.size()) == s.size());

  // A start position past the end finds nothing; it is not an error.
  assert(s.find("a", 99) == std::string::npos);
  assert(s.find('a', 99) == std::string::npos);
  assert(s.find("", 99) == std::string::npos);

  // A needle longer than what remains cannot match.
  assert(s.find("abcabcabc") == std::string::npos);
  assert(s.find("abc", 5) == std::string::npos);

  std::string t = "abc";
  assert(s.find(t) == 0);
  assert(s.find("abcXX", 0, 3) == 0);

  return 0;
}
