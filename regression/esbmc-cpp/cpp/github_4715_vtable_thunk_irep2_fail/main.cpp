// esbmc/esbmc#4715: the vtable type symbols are stored IREP2-side, so what the
// vtable value builder and the virtual-destructor dispatch read off a component
// has to be a field the IREP2 struct type carries. Dispatch through each base,
// and delete through the first, which is what reaches the destructor slot
// lookup.
#include <cassert>

struct A
{
  virtual ~A()
  {
  }
  virtual int a()
  {
    return 1;
  }
};

struct B
{
  virtual ~B()
  {
  }
  virtual int b()
  {
    return 2;
  }
};

struct C : A, B
{
  int a() override
  {
    return 10;
  }
  int b() override
  {
    return 20;
  }
};

int main()
{
  C c;
  A *pa = &c;
  B *pb = &c;
  assert(pa->a() == 10);
  assert(pb->b() == 21);

  A *heap = new C;
  assert(heap->a() == 10);
  delete heap;
  return 0;
}
