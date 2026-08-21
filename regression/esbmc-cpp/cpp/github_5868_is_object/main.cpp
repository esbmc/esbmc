#include <type_traits>
#include <cassert>
struct S {};
int main()
{
  static_assert(std::is_object<int>::value, "int is an object type");
  static_assert(std::is_object<S>::value, "class is an object type");
  static_assert(std::is_object<int *>::value, "pointer is an object type");
  static_assert(std::is_object<int[3]>::value, "array is an object type");
  static_assert(!std::is_object<void>::value, "void is not");
  static_assert(!std::is_object<int &>::value, "reference is not");
  static_assert(!std::is_object<int()>::value, "function is not");
  static_assert(std::is_object_v<int>, "_v alias");
  return 0;
}
