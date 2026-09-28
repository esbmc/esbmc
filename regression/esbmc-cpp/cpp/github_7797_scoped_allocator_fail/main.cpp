// #7797: the adaptor passes the inner allocator, not the outer one.
#include <scoped_allocator>
#include <cassert>

template <class T>
struct tagged_allocator
{
  typedef T value_type;
  int tag;

  tagged_allocator() : tag(0)
  {
  }
  explicit tagged_allocator(int t) : tag(t)
  {
  }
  template <class U>
  tagged_allocator(const tagged_allocator<U> &o) : tag(o.tag)
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

struct leading
{
  typedef tagged_allocator<int> allocator_type;
  int seen;

  leading(std::allocator_arg_t, const allocator_type &a, int) : seen(a.tag)
  {
  }
};

int main()
{
  std::scoped_allocator_adaptor<tagged_allocator<char>, tagged_allocator<int>>
    a(tagged_allocator<char>(1), tagged_allocator<int>(7));

  char storage[sizeof(leading)];
  leading *l = reinterpret_cast<leading *>(storage);
  a.construct(l, 5);
  assert(l->seen == 1);
  a.destroy(l);
  return 0;
}
