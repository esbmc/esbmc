// typeid refers to an object of static storage duration ([expr.typeid]/1); the
// comparison below is false, not a read of a destroyed temporary.
#include <typeinfo>
#include <cassert>

int main()
{
  const std::type_info *p = &typeid(int);
  assert(*p == typeid(char));
  return 0;
}
