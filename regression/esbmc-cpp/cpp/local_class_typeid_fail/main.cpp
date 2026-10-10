// Two local classes named S are different types ([class.local]), so their
// type_info objects compare unequal and the assertion fails.
#include <typeinfo>
#include <cassert>

const std::type_info &f()
{
  struct S
  {
  };
  return typeid(S);
}

const std::type_info &g()
{
  struct S
  {
  };
  return typeid(S);
}

int main()
{
  assert(f() == g());
  return 0;
}
