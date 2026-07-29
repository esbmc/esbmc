// github #6488: allocate_shared forwards its arguments to T's constructor and
// hands the result to the same reference-counted shared_ptr make_shared uses,
// so the object is released when the last owner dies. The allocator argument
// is accepted but not modelled.
#include <memory>
#include <cassert>

struct S
{
  int x;
  int y;
  S(int a, int b) : x(a), y(b)
  {
  }
};

int main()
{
  std::allocator<S> alloc;

  std::shared_ptr<S> p = std::allocate_shared<S, std::allocator<S>>(alloc, 3, 4);
  assert(p->x == 3);
  assert(p->y == 4);
  assert(p.use_count() == 1);

  {
    std::shared_ptr<S> q = p;
    assert(q.use_count() == 2);
    assert(q.get() == p.get());
  }
  assert(p.use_count() == 1); // the copy released its share

  std::shared_ptr<int> n =
    std::allocate_shared<int, std::allocator<int>>(std::allocator<int>());
  assert(*n == 0);

  return 0;
}
