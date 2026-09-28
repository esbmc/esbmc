// [allocator.traits.types]: allocator_traits exposes rebind_alloc/rebind_traits
// and the propagate_on_container_* traits. boost multi_index builds its node
// allocators through rebind_alloc, and src/util/symtab/context.h is a
// multi_index container -- so without it no ESBMC header reaching the symbol
// table parses. See #5868.
#include <memory>
#include <type_traits>
#include <cassert>

int main()
{
  typedef std::allocator<int> ai;
  typedef std::allocator_traits<ai> t;

  static_assert(
    std::is_same<t::rebind_alloc<char>, std::allocator<char> >::value,
    "rebind_alloc");
  static_assert(
    std::is_same<t::rebind_traits<char>::allocator_type, std::allocator<char> >::value,
    "rebind_traits");
  static_assert(std::is_same<t::void_pointer, void *>::value, "void_pointer");
  static_assert(
    std::is_same<t::const_void_pointer, const void *>::value,
    "const_void_pointer");
  static_assert(!t::propagate_on_container_swap::value, "pocs");
  static_assert(!t::propagate_on_container_copy_assignment::value, "pocca");

  ai a;
  ai b = t::select_on_container_copy_construction(a);
  int *p = t::allocate(b, 4);
  t::construct(b, p, 7);
  assert(p[0] == 7);
  t::deallocate(b, p, 4);
  return 0;
}
