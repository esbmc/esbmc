#include <memory>
#include <type_traits>
#include <cassert>

template <class T>
struct plain_alloc
{
  typedef T value_type;
  T *allocate(std::size_t n);
  void deallocate(T *p, std::size_t n);
};

int main()
{
  typedef std::allocator_traits<plain_alloc<int>>::rebind_alloc<char> rebound;
  // Rebinding rewrites the element type, so this is plain_alloc<char>.
  assert((std::is_same<rebound, plain_alloc<int>>::value));
  return 0;
}
