// A destructor and a replaced operator delete share one sideeffect2t::arguments
// vector -- slot 0 and slot 1. The IREP2 pass must fill slot 0 without
// disturbing slot 1, and must not decay the deallocator in slot 1 into the
// `&f` sugar: goto_convert's convert_cpp_delete then reads a pointer where it
// wants a code type and indexes an empty argument list
// (docs/roadmap/scope-clang-cpp-irep2.md §3.14, github #6494).
#include <cstddef>
#include <cassert>

static int deletes = 0;
static int dtors = 0;

void operator delete(void *) noexcept
{
  deletes++;
}
void operator delete(void *, size_t) noexcept
{
  deletes++;
}

struct C
{
  int v;
  C() : v(3)
  {
  }
  ~C()
  {
    dtors++;
  }
};

int main()
{
  C *p = new C();
  assert(p->v == 3);
  delete p;
  assert(dtors == 0 && deletes == 1);
  return 0;
}
