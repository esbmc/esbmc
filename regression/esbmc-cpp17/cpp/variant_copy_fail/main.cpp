// Non-vacuity guard for variant_copy: the copy really carries the alternative,
// so expecting the wrong index must FAIL.
#include <variant>
#include <cassert>

int main()
{
  std::variant<int, double> v = 2.5;
  std::variant<int, double> w = v;
  assert(w.index() == 0);
  return 0;
}
