// Anti-vacuity twin of allocator_traits_rebind: a rebound allocator really
// allocates its own value_type, so the storage it hands back holds what was
// constructed into it.
#include <memory>
#include <cassert>

int main()
{
  typedef std::allocator_traits<std::allocator<int> > t;
  t::rebind_alloc<char> ca;
  typedef std::allocator_traits<t::rebind_alloc<char> > ct;
  char *p = ct::allocate(ca, 4);
  ct::construct(ca, p, 'x');
  assert(p[0] == 'y');
  ct::deallocate(ca, p, 4);
  return 0;
}
