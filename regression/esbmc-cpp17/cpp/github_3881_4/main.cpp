// Test: operator bool() for unique_ptr — null and non-null cases
#include <cassert>
#include <memory>

int main()
{
  std::unique_ptr<int> empty;
  assert(!empty);          // null unique_ptr is false
  assert(empty.get() == nullptr);

  std::unique_ptr<int> p = std::make_unique<int>(7);
  assert(p);               // non-null unique_ptr is true
  assert(*p == 7);

  // After release, becomes null
  int *raw = p.release();
  assert(!p);
  assert(*raw == 7);
  delete raw;

  return 0;
}
