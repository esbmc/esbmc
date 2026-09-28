#include <type_traits>
#include <cassert>

struct S
{
  void fn();
};

int main()
{
  // [meta.unary.cat]: a pointer to member function is not an object pointer.
  assert((std::is_member_object_pointer<void (S::*)()>::value));
  return 0;
}
