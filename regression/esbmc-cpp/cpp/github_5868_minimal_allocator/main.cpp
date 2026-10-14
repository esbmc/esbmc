#include <memory>
#include <cassert>
#include <cstdlib>

// [allocator.requirements] requires only these three members.
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

  assert(tr::max_size(a) == std::size_t(-1) / sizeof(int));

  int *p = tr::allocate(a, 1);
  if (p == NULL)
    return 0;
  tr::construct(a, p, 42);
  assert(*p == 42);
  tr::destroy(a, p);
  tr::deallocate(a, p, 1);

  // An allocator that does supply the members keeps using them.
  std::allocator<int> sa;
  int *q = std::allocator_traits<std::allocator<int>>::allocate(sa, 1);
  if (q == NULL)
    return 0;
  std::allocator_traits<std::allocator<int>>::construct(sa, q, 7);
  assert(*q == 7);
  std::allocator_traits<std::allocator<int>>::destroy(sa, q);
  std::allocator_traits<std::allocator<int>>::deallocate(sa, q, 1);
  return 0;
}
