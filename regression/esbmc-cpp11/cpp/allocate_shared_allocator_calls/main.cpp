// github #6488, KNOWNBUG: pins the one thing the allocate_shared model gets
// wrong. [util.smartptr.shared.create] requires the object's storage to come
// from the supplied allocator, so g_allocs is 1 here -- g++ builds this and it
// exits 0. ESBMC's <memory> routes allocate_shared through `new` instead
// (shared_ptr releases with `delete`, so allocator-supplied malloc'd storage
// would report a deallocation-operator mismatch), so allocate() is never
// called, g_allocs stays 0, and ESBMC reports FAILED.
//
// This is the unsound direction: the same gap lets ESBMC *prove*
// `g_allocs == 0`, a property false of every conforming implementation. Flip
// this test to CORE when allocators are modelled for real.
#include <memory>
#include <cassert>
#include <cstdlib>

static int g_allocs = 0;

template <typename T>
struct CountingAlloc
{
  typedef T value_type;
  CountingAlloc()
  {
  }
  template <typename U>
  CountingAlloc(const CountingAlloc<U> &)
  {
  }
  T *allocate(size_t n)
  {
    ++g_allocs;
    return (T *)malloc(n * sizeof(T));
  }
  void deallocate(T *p, size_t)
  {
    free(p);
  }
};

struct S
{
  int x;
  S(int v) : x(v)
  {
  }
};

int main()
{
  CountingAlloc<S> a;
  std::shared_ptr<S> p = std::allocate_shared<S, CountingAlloc<S>>(a, 7);
  assert(p->x == 7);
  assert(g_allocs == 1);
  return 0;
}
