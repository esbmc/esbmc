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
  std::deque<int, tracking_allocator<int> > e;
  e.push_back(6);
  e.push_back(7);

  assert(e.front() == 7);
  return 0;
}
