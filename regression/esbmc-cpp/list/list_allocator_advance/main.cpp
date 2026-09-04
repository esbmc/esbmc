#include <list>
#include <cstddef>
#include <iterator>
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

/* Before C++11 <iterator>'s generic advance is `i = i + n`, which a
   bidirectional list iterator cannot do, so <list>'s own overload is the
   fallback. It must accept a list with a non-default allocator. */
int main()
{
  std::list<int, tracking_allocator<int> > m;
  m.push_back(1);
  m.push_back(2);
  m.push_back(3);

  std::list<int, tracking_allocator<int> >::iterator it = m.begin();
  std::advance(it, 2);
  assert(*it == 3);
  return 0;
}
