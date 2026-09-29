// #7797: "(a)(b)c" has two marked sub-expressions, not three.
#include <regex>
#include <cassert>

int main()
{
  std::regex re("(a)(b)c");
  assert(re.mark_count() == 3);
  return 0;
}
