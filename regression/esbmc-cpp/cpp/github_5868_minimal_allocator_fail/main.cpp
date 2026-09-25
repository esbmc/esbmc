#include <memory>
#include <cassert>
#include <cstdlib>

template <class T>
struct minimal_alloc
{
  typedef T value_type;
  T *allocate(std::size_t n)
  {
    return static_cast<T *>(malloc(sizeof(T) * n));
  }
  void deallocate(T *p, std::size_t)
  {
    free(p);
  }
};

int main()
{
  typedef std::allocator_traits<minimal_alloc<int>> tr;
  minimal_alloc<int> a;
  int *p = tr::allocate(a, 1);
  if (p == NULL)
    return 1;
  tr::construct(a, p, 42);
  // construct copy-initialises from the argument, so this is 42, not 0.
  assert(*p == 0);
  return 0;
}
