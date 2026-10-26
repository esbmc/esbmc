#include <memory>
#include <cassert>

struct H
{
  std::shared_ptr<int> p;
};

int main()
{
  H a{std::make_shared<int>(1)};
  H b = a;
  assert(b.p.use_count() == 1);
  return 0;
}
