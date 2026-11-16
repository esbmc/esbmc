#include <cassert>
#include <memory>

int main()
{
  std::shared_ptr<int> a(new int(1));
  std::shared_ptr<int> b = a;
  a.reset();
  assert(a.get() == nullptr && !a && a.use_count() == 0);
  assert(*b == 1 && b.use_count() == 1);

  std::shared_ptr<int> c = b;
  b = std::shared_ptr<int>();
  assert(b.get() == nullptr && *c == 1);

  std::shared_ptr<int> d = c;
  c.reset(new int(2));
  assert(*c == 2 && *d == 1 && d.use_count() == 1);
  return 0;
}
