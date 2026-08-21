#include <optional>
#include <cassert>

int main()
{
  std::optional<int> e;
  // [optional.comp.with.t]: a disengaged optional compares unequal to any
  // value, so this must be refuted.
  assert(e == 3);
  return 0;
}
