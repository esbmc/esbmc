#include <any>
#include <cassert>

int main()
{
  std::any a = 5;

  // Active type: pointer is non-null and points to the stored value.
  if (auto *p = std::any_cast<int>(&a))
    assert(*p == 5);
  else
    assert(false);

  // Inactive type: any_cast<T*> returns nullptr.
  assert(std::any_cast<double>(&a) == nullptr);

  // Empty any: any_cast<T*> returns nullptr regardless of T.
  std::any empty;
  assert(std::any_cast<int>(&empty) == nullptr);

  return 0;
}
