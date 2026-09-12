// The two shapes the destructor-attaching arm must decline: a pointer to a
// non-class type, and a class with no destructor at all. Declining leaves
// `delete` exactly as it was; not declining dereferences a null class type or
// a null destructor component (docs/roadmap/scope-clang-cpp-irep2.md §3.14).
#include <cassert>

struct S
{
  int x;
};

int main()
{
  S *s = new S();
  s->x = 7;
  const int v = s->x;
  delete s;

  int *p = new int(5);
  const int w = *p;
  delete p;

  assert(v == 7 && w == 5);
}
