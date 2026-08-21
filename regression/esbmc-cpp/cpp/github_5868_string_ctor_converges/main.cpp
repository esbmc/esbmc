#include <string>
#include <filesystem>
#include <cassert>

int main()
{
  // A conditional source defeats the NUL-scan folding that a plain literal
  // gets, so this constructor used not to converge.
  const char *x = "abc";
  std::string s(x ? x : "");
  assert(s.size() == 3);
  assert(s[0] == 'a');
  assert(s[2] == 'c');

  std::string e(x ? "" : x);
  assert(e.size() == 0);
  assert(e.empty());

  // <filesystem>'s path is built exactly that way.
  std::filesystem::path p("a/b");
  assert(!p.empty());
  return 0;
}
