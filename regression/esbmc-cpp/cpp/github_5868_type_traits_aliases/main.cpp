#include <type_traits>
#include <cassert>

// [meta.trans.cv] / [meta.trans.ref] / [meta.trans.arr]: the _t aliases.
static_assert(std::is_same<std::add_const_t<int>, const int>::value, "");
static_assert(std::is_same<std::add_volatile_t<int>, volatile int>::value, "");
static_assert(std::is_same<std::add_cv_t<int>, const volatile int>::value, "");
static_assert(std::is_same<std::add_lvalue_reference_t<int>, int &>::value, "");
static_assert(std::is_same<std::add_rvalue_reference_t<int>, int &&>::value, "");
static_assert(std::is_same<std::remove_all_extents_t<int>, int>::value, "");
static_assert(std::is_same<std::remove_all_extents_t<int[]>, int>::value, "");
static_assert(std::is_same<std::remove_all_extents_t<int[2][3]>, int>::value, "");

template <class T>
int width()
{
  return sizeof(std::remove_all_extents_t<T>);
}

int main()
{
  std::add_cv_t<int> a = 3;
  assert(a == 3);
  assert(width<char[4][5]>() == 1);
  assert(width<int[2]>() == sizeof(int));
  return 0;
}
