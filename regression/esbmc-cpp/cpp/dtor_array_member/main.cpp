// C++ [class.dtor]/9: an array member of class type must have each element
// destroyed in reverse index order. build_destructor_chain in
// clang_cpp_convert.cpp now handles array members (see the isolated,
// passing dtor_array_member_heap test), so ~C destroys a[2], a[1], a[0].
//
// This stack-scoped variant still fails for two orthogonal, pre-existing
// reasons the destructor-chain fix does not touch:
//   1. B is empty, so assigning the reconstituted `C c = tmp$1` value trips
//      "assignment to constant_array not handled" in goto-symex (constant
//      array LHS is not projected like constant_struct/constant_union).
//   2. A local `C c;` is materialised through a temporary that is then also
//      destroyed, so the element destructors would run twice (g == 6).
// Once both are addressed this test should assert g == 3 cleanly.
#include <cassert>

int g;

struct B
{
  ~B() { g++; }
};

struct C
{
  B a[3];
};

int main()
{
  {
    C c;
  }
  assert(g == 3);
  return 0;
}
