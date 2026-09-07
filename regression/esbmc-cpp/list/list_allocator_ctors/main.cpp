#include <list>
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
typedef std::list<int, alloc> ilist;

int main()
{
  alloc a;

  /* [list.cons]: every constructor has an allocator-taking form. */
  ilist empty(a);
  assert(empty.size() == 0);

  ilist n_default(2, a);
  assert(n_default.size() == 2);
  assert(n_default.front() == 0);

  ilist n_value(3, 7, a);
  assert(n_value.size() == 3);
  assert(n_value.front() == 7);
  assert(n_value.back() == 7);

  ilist copied(n_value, a);
  assert(copied.size() == 3);
  assert(copied.front() == 7);

  int raw[2] = {4, 5};
  ilist from_pointers(raw, raw + 2, a);
  assert(from_pointers.size() == 2);
  assert(from_pointers.front() == 4);

  ilist from_iterators(from_pointers.begin(), from_pointers.end(), a);
  assert(from_iterators.size() == 2);
  assert(from_iterators.back() == 5);

  ilist from_init({8, 9}, a);
  assert(from_init.size() == 2);
  assert(from_init.front() == 8);

  return 0;
}
