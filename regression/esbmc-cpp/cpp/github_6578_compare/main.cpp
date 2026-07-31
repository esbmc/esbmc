// The bundled comparison categories declare the defaulted friend form, so
// equality between category values exercised the same defect.
#include <compare>
#include <cassert>

int main()
{
  assert(std::strong_ordering::less == std::strong_ordering::less);
  assert(!(std::strong_ordering::less == std::strong_ordering::greater));
  assert(!(std::partial_ordering::unordered == std::partial_ordering::less));
  return 0;
}
