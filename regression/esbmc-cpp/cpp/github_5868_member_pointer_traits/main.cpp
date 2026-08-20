#include <type_traits>

struct S
{
  int data;
  void fn();
  int cfn() const;
};

int main()
{
  // [meta.unary.cat]: a pointer to member is object-or-function, never both.
  static_assert(std::is_member_object_pointer<int S::*>::value, "data ptr");
  static_assert(!std::is_member_function_pointer<int S::*>::value, "data ptr");

  static_assert(std::is_member_function_pointer<void (S::*)()>::value, "fn ptr");
  static_assert(!std::is_member_object_pointer<void (S::*)()>::value, "fn ptr");
  static_assert(
    std::is_member_function_pointer<int (S::*)() const>::value, "const fn ptr");

  static_assert(!std::is_member_object_pointer<int>::value, "plain int");
  static_assert(!std::is_member_function_pointer<int>::value, "plain int");
  static_assert(!std::is_member_object_pointer<int *>::value, "plain pointer");

  // cv-qualified member pointers still classify.
  static_assert(
    std::is_member_object_pointer<int S::*const>::value, "const data ptr");

  static_assert(std::is_member_object_pointer_v<int S::*>, "_v alias");
  static_assert(std::is_member_function_pointer_v<void (S::*)()>, "_v alias");
  return 0;
}
