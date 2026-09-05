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

int main()
{
  std::deque<int, std::allocator<int> > d;
  d.push_back(7);
  d.push_front(6);
  assert(d.size() == 2);
  assert(d.front() == 6);
  assert(d.back() == 7);

  std::deque<int, tracking_allocator<int> > e;
  e.push_back(6);
  e.push_back(7);
  assert(e.size() == 2);
  assert(e[0] == 6);
  assert(e[1] == 7);

  std::deque<int, tracking_allocator<int> > f;
  f.push_back(6);
  f.push_back(7);
  assert(e == f);
  assert(!(e < f));

  /* [deque.overview]: get_allocator() returns allocator_type. */
  tracking_allocator<int> a = e.get_allocator();
  std::allocator<int> b = d.get_allocator();
  (void)a;
  (void)b;

  return 0;
}
