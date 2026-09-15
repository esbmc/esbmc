// std::pmr core. Every expectation was first compiled and run under
// g++ -std=c++17 against real libstdc++.
#include <memory_resource>
#include <new>
#include <cassert>

int main()
{
  std::pmr::memory_resource *r = std::pmr::new_delete_resource();
  assert(r != nullptr);
  assert(std::pmr::get_default_resource() == r);

  std::pmr::polymorphic_allocator<int> a(r);
  assert(a.resource() == r);

  int *p = a.allocate(4);
  assert(p != nullptr);
  p[0] = 11;
  p[3] = 22;
  assert(p[0] == 11 && p[3] == 22);
  a.deallocate(p, 4);

  std::pmr::polymorphic_allocator<int> b;
  assert(b.resource() == std::pmr::get_default_resource());
  assert(a == b);

  std::pmr::memory_resource *old =
    std::pmr::set_default_resource(std::pmr::null_memory_resource());
  assert(old == r);
  assert(std::pmr::get_default_resource() == std::pmr::null_memory_resource());

  bool threw = false;
  try
  {
    void *q = std::pmr::null_memory_resource()->allocate(1);
    (void)q;
  }
  catch (const std::bad_alloc &)
  {
    threw = true;
  }
  assert(threw);

  std::pmr::set_default_resource(old);
  assert(std::pmr::get_default_resource() == r);

  assert(*r == *std::pmr::new_delete_resource());
  assert(*r != *std::pmr::null_memory_resource());
  return 0;
}
