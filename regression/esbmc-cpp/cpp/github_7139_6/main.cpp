#include <cassert>
#include <string>

int main()
{
  // Deliberately unbounded: the grow loop has to converge on its own, without
  // --unwind or --incremental-bmc bounding it.
  std::string s("abc");
  s.resize(5);
  assert(s.size() == 5);
  assert(s[4] == '\0');

  std::string t("abc");
  t.resize(5, 'x');
  assert(t.size() == 5);
  assert(t[4] == 'x');

  return 0;
}
