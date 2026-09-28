#include <cassert>
#include <type_traits>

// The negative direction of the traits added for github #5868 gap 1: a trait
// hardcoded to true would pass type_traits_layout_family and fail nothing.
struct WithVirt
{
  virtual ~WithVirt();
  int a;
};

int main()
{
  assert(std::is_standard_layout<WithVirt>::value);
  return 0;
}
