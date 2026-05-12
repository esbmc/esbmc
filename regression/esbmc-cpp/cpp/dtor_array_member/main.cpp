// C++ [class.dtor]/9: an array member of class type must have each element
// destroyed in reverse index order. The build_destructor_chain in
// clang_cpp_convert.cpp currently skips array members because
// QualType::getAsCXXRecordDecl returns null for array types. Once array
// members are handled, ~C will increment g three times and this test will
// start passing.
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
