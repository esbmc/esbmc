#include <string>
#include <cassert>

int main()
{
  std::string s("abc");
  s.clear();
  // clear() erases every character, so size() is 0.
  assert(s.size() == 3);
  return 0;
}
