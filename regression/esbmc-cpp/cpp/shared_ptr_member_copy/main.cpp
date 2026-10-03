// Copying a class that holds a shared_ptr member keeps the control block
// alive. The aggregate initialisation used to destroy the temporary
// shared_ptr as well as the member, a second __release that freed the shared
// object while an owner still referred to it. g++ runs this program with the
// assertion holding.
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
