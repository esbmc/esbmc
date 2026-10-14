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

int main()
{
  std::list<int, std::allocator<int> > l;
  l.push_back(1);
  l.push_back(2);
  assert(l.size() == 2);
  assert(l.front() == 1);
  assert(l.back() == 2);

  std::list<int, tracking_allocator<int> > m;
  m.push_back(1);
  m.push_back(2);

  int sum = 0;
  for (std::list<int, tracking_allocator<int> >::iterator it = m.begin();
       it != m.end();
       ++it)
    sum += *it;
  assert(sum == 3);

  std::list<int, tracking_allocator<int> > n;
  n.push_back(1);
  n.push_back(2);
  assert(m == n);
  assert(!(m < n));

  /* [list.overview]: get_allocator() returns allocator_type. */
  tracking_allocator<int> a = m.get_allocator();
  std::allocator<int> b = l.get_allocator();
  (void)a;
  (void)b;

  return 0;
}
