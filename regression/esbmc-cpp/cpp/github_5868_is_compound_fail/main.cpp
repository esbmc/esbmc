#include <type_traits>
#include <cassert>

int main()
{
  // [meta.unary.comp]: int is fundamental, so it is not compound.
  assert(std::is_compound<int>::value);
  return 0;
}
