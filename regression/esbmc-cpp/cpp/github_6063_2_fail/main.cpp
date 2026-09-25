#include <memory>
#include <cassert>

int main()
{
  std::allocator<int> a;
  typedef std::allocator_traits<std::allocator<int>> AT;
  int *p = AT::allocate(a, 1);
  AT::construct(a, p, 7);
  assert(*p == 8); // must fail: construct stored 7
  AT::deallocate(a, p, 1);
  return 0;
}
