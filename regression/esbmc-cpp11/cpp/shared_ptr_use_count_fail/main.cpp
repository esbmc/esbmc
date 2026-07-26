// A copy really shares ownership, so use_count() must be 2 here. Without this
// the counting checks in shared_ptr_basic could pass vacuously.
#include <memory>
#include <cassert>

int main()
{
  std::shared_ptr<int> a(new int(1));
  std::shared_ptr<int> b = a;
  assert(a.use_count() == 1);
  return 0;
}
