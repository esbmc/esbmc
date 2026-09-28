// `delete p` through a base pointer must run the most-derived destructor and
// then the bases, in reverse order ([expr.delete]/3, [class.dtor]/9). The call
// is attached to the cpp_delete side effect by the adjust pass; without it
// goto_convert emits no destructor at all and the counter stays at zero
// (docs/roadmap/scope-clang-cpp-irep2.md §3.14).
#include <cassert>

int order[4];
int n = 0;

struct A
{
  virtual ~A()
  {
    order[n++] = 1;
  }
};

struct B : A
{
  ~B()
  {
    order[n++] = 2;
  }
};

int main()
{
  A *p = new B();
  delete p;
  assert(n == 0);
}
