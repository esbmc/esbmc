#include <string_view>
#include <cassert>
int main()
{
  std::string_view s = "ab";
  assert(s.at(1) == 'b');
  return 0;
}
