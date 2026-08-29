#include <type_traits>
#include <algorithm>

int main()
{
  // [meta.trans.cv]
  static_assert(
    std::is_same<std::add_cv<int>::type, const volatile int>::value, "add_cv");
  static_assert(
    std::is_same<std::add_volatile<int>::type, volatile int>::value,
    "add_volatile");
  static_assert(
    std::is_same<std::add_cv<const int>::type, const volatile int>::value,
    "add_cv is idempotent on an already-const type");
  static_assert(
    std::is_same<std::add_volatile<int &>::type, int &>::value,
    "a reference is left alone");

  // [alg.move]. Taking its address instantiates the template. Calling it
  // cannot be checked here: the copy_backward it mirrors does not converge
  // under symbolic execution either, so a call would only time out.
  int *(*f)(int *, int *, int *) = &std::move_backward<int *, int *>;
  (void)f;
  return 0;
}
