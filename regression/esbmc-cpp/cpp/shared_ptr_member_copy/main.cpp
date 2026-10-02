// Copying a class that holds a shared_ptr member. The member of `a` used to be
// built in a temporary that was copied in bitwise and then destroyed, releasing
// a reference `a` still held, so the copy read a freed control block.
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
  assert(*b.p == 1);
  return 0;
}
