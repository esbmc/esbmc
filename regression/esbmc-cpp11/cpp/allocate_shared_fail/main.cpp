// github #6488: the negative side of allocate_shared_basic -- the constructor
// arguments really are forwarded, so a claim about the wrong value is refuted
// rather than vacuously proved.
#include <memory>
#include <cassert>

struct S
{
  int x;
  S(int v) : x(v)
  {
  }
};

int main()
{
  std::shared_ptr<S> p =
    std::allocate_shared<S, std::allocator<S>>(std::allocator<S>(), 3);
  assert(p->x == 4);
  return 0;
}
