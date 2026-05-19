// C++ [class.dtor]/9: the most-derived destructor must run each virtual
// base's destructor exactly once, in reverse order. The build_destructor_chain
// in clang_cpp_convert.cpp currently skips virtual bases unconditionally
// because ESBMC does not yet split D1 (complete-object) and D2 (base-object)
// destructors. Once that split exists, ~B will run for the virtual base and
// this test will start passing.
#include <cassert>

int g;

struct B
{
  ~B() { g = 42; }
};

struct D : virtual B
{
};

int main()
{
  {
    D d;
  }
  assert(g == 42);
  return 0;
}
