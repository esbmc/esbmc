#include <cassert>
#include <variant>

int main()
{
  std::variant<int, double> v = 5;

  // Active alternative: pointer is non-null and points to the stored value.
  if (auto *p = std::get_if<int>(&v))
    assert(*p == 5);
  else
    assert(false);

  // Inactive alternative: get_if returns nullptr.
  assert(std::get_if<double>(&v) == nullptr);

  // After reassignment, get_if flips.
  v = 2.5;
  assert(std::get_if<int>(&v) == nullptr);
  assert(std::get_if<double>(&v) != nullptr);
  assert(*std::get_if<double>(&v) == 2.5);

  return 0;
}
