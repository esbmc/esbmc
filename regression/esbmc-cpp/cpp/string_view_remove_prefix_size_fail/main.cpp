// Anti-vacuity twin of string_view_remove_prefix_size: the size has to shrink
// by exactly n, not by some other amount.
#include <string_view>
#include <cassert>

int main()
{
  std::string_view sv("hello world");
  sv.remove_prefix(6);
  assert(sv.size() == 11);
  return 0;
}
