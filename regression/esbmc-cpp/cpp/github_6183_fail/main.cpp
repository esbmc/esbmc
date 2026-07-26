#include <cassert>
#include <memory>

int main()
{
  std::unique_ptr<int> p(new int(7));
  p = nullptr;
  // Must fail: assigning nullptr releases the pointee and stores null.
  assert(p.get() != nullptr);
  return 0;
}
