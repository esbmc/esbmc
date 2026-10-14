// C++ [class.dtor]/9: an array member of class type must have each element
// destroyed. build_destructor_chain in clang_cpp_convert.cpp used to skip
// array members because QualType::getAsCXXRecordDecl returns null for array
// types, so ~C had an empty body and no ~B ran. Now every element is
// destroyed, so ~C runs ~B once per element (three times here).
//
// A heap object (new/delete) is used so the destructor is invoked exactly
// once, isolating the per-element member-destruction behaviour.
#include <cassert>

int g;

struct B
{
  int x;
  ~B()
  {
    g++;
  }
};

struct C
{
  B a[3];
};

int main()
{
  C *c = new C();
  delete c;
  assert(g == 3);
  return 0;
}
