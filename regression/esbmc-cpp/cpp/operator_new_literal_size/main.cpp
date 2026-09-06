// The control for heap_size_through_parameter: the same allocation and the
// same in-bounds writes, with the size a literal inside the callee. This one
// verifies, which is what localises the defect to the size being a parameter.
#include <new>
#include <cassert>

struct R
{
  virtual void *alloc()
  {
    return ::operator new(16);
  }
  virtual ~R()
  {
  }
};

int main()
{
  R r;
  int *p = static_cast<int *>(r.alloc());
  p[0] = 11;
  p[3] = 22;
  assert(p[0] == 11 && p[3] == 22);
  ::operator delete(p);
  return 0;
}
