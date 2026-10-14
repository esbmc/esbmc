// github #6494: the aligned deallocation form takes two parameters like the
// C++14 sized form, but wants a std::align_val_t rather than the byte count
// this lowering supplies. Routing it passed sizeof(C) where alignof(C) belongs,
// so pin that the replacement is either called with the right alignment or
// left on the built-in path -- never called with the object's size.
#include <cstddef>
#include <new>
#include <cassert>

static char pool[512];
static size_t seen = 0;

struct alignas(32) C
{
  char buf[128];

  static void *operator new(size_t, std::align_val_t)
  {
    return pool;
  }
  static void operator delete(void *, std::align_val_t a)
  {
    seen = (size_t)a;
  }
};

int main()
{
  C *p = new C();
  delete p;
  assert(seen == 0 || seen == 32); // never sizeof(C), which is 128
  return 0;
}
