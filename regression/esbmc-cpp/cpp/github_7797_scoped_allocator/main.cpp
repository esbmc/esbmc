// #7797: <scoped_allocator> must be includable, and the adaptor must pass its
// inner allocator to an element that uses one.
#include <scoped_allocator>
#include <cassert>
#include <memory>

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
  bool operator==(const tagged_allocator &o) const
  {
    return tag == o.tag;
  }
};

// Takes the allocator after allocator_arg.
struct leading
{
  typedef tagged_allocator<int> allocator_type;
  int seen, value;

  leading(std::allocator_arg_t, const allocator_type &a, int v)
    : seen(a.tag), value(v)
  {
  }
};

// Takes the allocator as its last argument.
struct trailing
{
  typedef tagged_allocator<int> allocator_type;
  int seen, value;

  trailing(int v, const allocator_type &a) : seen(a.tag), value(v)
  {
  }
};

struct plain
{
  int value;
  explicit plain(int v) : value(v)
  {
  }
};

int main()
{
  typedef std::scoped_allocator_adaptor<
    tagged_allocator<char>,
    tagged_allocator<int>>
    adaptor;

  adaptor a(tagged_allocator<char>(1), tagged_allocator<int>(7));
  assert(a.outer_allocator().tag == 1);
  assert(a.inner_allocator().outer_allocator().tag == 7);

  assert((std::uses_allocator<leading, tagged_allocator<int>>::value));
  assert((std::uses_allocator<trailing, tagged_allocator<int>>::value));
  assert((!std::uses_allocator<plain, tagged_allocator<int>>::value));

  char storage[sizeof(leading)];
  leading *l = reinterpret_cast<leading *>(storage);
  a.construct(l, 5);
  assert(l->seen == 7 && l->value == 5);
  a.destroy(l);

  char storage2[sizeof(trailing)];
  trailing *t = reinterpret_cast<trailing *>(storage2);
  a.construct(t, 6);
  assert(t->seen == 7 && t->value == 6);
  a.destroy(t);

  char storage3[sizeof(plain)];
  plain *p = reinterpret_cast<plain *>(storage3);
  a.construct(p, 9);
  assert(p->value == 9);
  a.destroy(p);

  char *raw = a.allocate(4);
  raw[3] = 'x';
  assert(raw[3] == 'x');
  a.deallocate(raw, 4);

  std::scoped_allocator_adaptor<tagged_allocator<int>> single(
    tagged_allocator<int>(3));
  assert(single.outer_allocator().tag == 3);
  assert(single.inner_allocator().outer_allocator().tag == 3);
  return 0;
}
