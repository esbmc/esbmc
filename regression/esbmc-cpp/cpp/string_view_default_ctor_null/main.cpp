// [string.view.cons]: the default constructor gives data() == nullptr and
// size() == 0. The model set only the size, leaving the pointer indeterminate,
// so data() on a default-constructed view returned garbage.
#include <string_view>
#include <cassert>

int main()
{
  std::string_view sv;
  assert(sv.size() == 0);
  assert(sv.empty());
  assert(sv.data() == nullptr);
  assert(sv.begin() == sv.end());
  return 0;
}
