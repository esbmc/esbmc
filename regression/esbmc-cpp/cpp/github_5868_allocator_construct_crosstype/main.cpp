#include <memory>
#include <cassert>
#include <cstdlib>

struct node
{
  int payload;
};

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
  // [allocator.traits.members]: construct/destroy are templates on the pointee,
  // so a traits instantiated on a node allocator can build an int -- the shape
  // boost multi_index uses.
  typedef std::allocator_traits<minimal_alloc<node>> tr;
  minimal_alloc<node> a;

  int *p = static_cast<int *>(malloc(sizeof(int)));
  if (p == NULL)
    return 0;
  tr::construct(a, p, 42);
  assert(*p == 42);
  tr::destroy(a, p);
  free(p);

  node *q = tr::allocate(a, 1);
  if (q == NULL)
    return 0;
  tr::construct(a, q);
  tr::destroy(a, q);
  tr::deallocate(a, q, 1);
  return 0;
}
