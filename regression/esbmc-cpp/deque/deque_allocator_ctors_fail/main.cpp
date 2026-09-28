#include <deque>
#include <cstddef>
#include <memory>
#include <cassert>

/* A minimal conforming allocator. ESBMC's container models never call
   allocate/deallocate -- the allocator is carried for type identity only. */
template <class T>
struct tracking_allocator
{
  typedef T value_type;
  tracking_allocator()
  {
  }
  template <class U>
  tracking_allocator(const tracking_allocator<U> &)
  {
  }
  T *allocate(std::size_t n)
  {
    return static_cast<T *>(::operator new(n * sizeof(T)));
  }
  void deallocate(T *p, std::size_t)
  {
    ::operator delete(p);
  }
};

typedef tracking_allocator<int> alloc;
typedef std::deque<int, alloc> ideque;

int main()
{
  alloc a;

  /* [deque.cons]: every constructor has an allocator-taking form. */
  ideque empty(a);
  assert(empty.size() == 0);

  ideque n_default(2, a);
  assert(n_default.size() == 2);
  assert(n_default[0] == 0);

  ideque n_value(3, 7, a);
  assert(n_value.size() == 3);
  assert(n_value[0] == 8);
  assert(n_value[2] == 7);

  ideque copied(n_value, a);
  assert(copied.size() == 3);
  assert(copied[1] == 7);

  ideque from_init({8, 9}, a);
  assert(from_init.size() == 2);
  assert(from_init[0] == 8);

  return 0;
}
