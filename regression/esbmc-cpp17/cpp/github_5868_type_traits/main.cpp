// github #5868 gap 7: <type_traits> was missing several classification traits
// and most of the C++14 _t aliases, and <tuple> had tuple_element but no
// tuple_element_t. TMP-heavy headers (llvm/ADT, fmt) name these, so their
// absence blocks the ~24-TU clang-frontend family from parsing.
//
// Everything here is compile-time: the value of the test is that it compiles.
#include <type_traits>
#include <tuple>
#include <utility>
#include <cassert>

struct A
{
};
struct B : A
{
};
union U
{
  int i;
};
struct P
{
  virtual ~P()
  {
  }
};
struct Abstract
{
  virtual void f() = 0;
};
struct Final final
{
};

static_assert(std::is_base_of<A, B>::value, "is_base_of");
static_assert(std::is_class<A>::value, "is_class");
static_assert(std::is_union<U>::value, "is_union");
static_assert(!std::is_union<A>::value, "is_union negative");
static_assert(std::is_polymorphic<P>::value, "is_polymorphic");
static_assert(std::is_abstract<Abstract>::value, "is_abstract");
static_assert(!std::is_abstract<A>::value, "is_abstract negative");
static_assert(std::is_empty<A>::value, "is_empty");
static_assert(!std::is_empty<P>::value, "is_empty negative");
static_assert(std::is_final<Final>::value, "is_final");
static_assert(!std::is_final<A>::value, "is_final negative");
static_assert(std::is_member_pointer<int A::*>::value, "is_member_pointer");

static_assert(std::is_same<std::remove_const_t<const int>, int>::value, "1");
static_assert(
  std::is_same<std::remove_volatile_t<volatile int>, int>::value,
  "2");
static_assert(std::is_same<std::remove_pointer_t<int *>, int>::value, "3");
static_assert(std::is_same<std::add_const_t<int>, const int>::value, "4");
static_assert(std::is_same<std::add_pointer_t<int>, int *>::value, "5");
static_assert(
  std::is_same<std::add_lvalue_reference_t<int>, int &>::value,
  "6");
static_assert(std::is_same<std::common_type_t<int, long>, long>::value, "7");
static_assert(
  std::is_same<std::remove_cv_t<const volatile int>, int>::value,
  "8");

static_assert(
  std::is_same<std::tuple_element_t<0, std::tuple<int, char>>, int>::value,
  "9");
static_assert(
  std::is_same<std::tuple_element_t<1, std::tuple<int, char>>, char>::value,
  "10");
static_assert(
  std::is_same<std::tuple_element_t<1, std::pair<int, char>>, char>::value,
  "11");

int main()
{
  // One runtime check so the test is not purely a compile gate.
  std::tuple_element_t<0, std::tuple<int, char>> v = 7;
  assert(v == 7);
  return 0;
}
