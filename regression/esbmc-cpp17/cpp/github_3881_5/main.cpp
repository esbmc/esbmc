// Test: unique_ptr::swap() — exchange ownership between two unique_ptrs
#include <cassert>
#include <memory>

int main()
{
  std::unique_ptr<int> a = std::make_unique<int>(1);
  std::unique_ptr<int> b = std::make_unique<int>(2);

  a.swap(b);
  assert(*a == 2);
  assert(*b == 1);

  // Non-member swap via ADL
  swap(a, b);
  assert(*a == 1);
  assert(*b == 2);

  return 0;
}
