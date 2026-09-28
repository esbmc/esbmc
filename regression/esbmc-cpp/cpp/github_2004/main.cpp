// esbmc/esbmc#2004: `operator new(n)` with a raw byte count (not a sizeof)
// used to abort with type2t::symbolic_type_excp because the result was a
// dereference of a void pointer. It must now allocate n raw bytes and return
// a usable pointer.
#include <cstdlib>
#include <cassert>

struct aaaa
{
  int a;
  int b;
};

int main()
{
  void *foo = operator new(8);
  aaaa *p = static_cast<aaaa *>(foo);
  p->a = 5;
  p->b = 7;
  assert(p->a + p->b == 12);
  operator delete(foo);
  return 0;
}
