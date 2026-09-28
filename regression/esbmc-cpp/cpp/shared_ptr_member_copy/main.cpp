// KNOWNBUG. Copying a class that holds a shared_ptr member reports
// "invalidated dynamic object" inside shared_ptr's __release: the implicit
// copy constructor does not keep the control block alive, so the shared
// object is treated as freed while a live owner still refers to it.
// g++ runs this program with the assertion holding.
//
// It is the copy of the WRAPPER, not shared_ptr itself -- shared_ptr_copy_controls
// pins the neighbouring shapes that work: copying a shared_ptr directly,
// returning a wrapper by value from a function, and returning a bare
// shared_ptr.
//
// Any class holding a shared_ptr by value hits this, which is the ordinary way
// shared ownership is written.
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
