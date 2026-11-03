#include <cassert>
#include <memory>

int main()
{
  std::shared_ptr<int> a(new int(1));
  std::shared_ptr<int> b = a;
  a.reset();
  assert(a.get() != nullptr);
  return 0;
}
