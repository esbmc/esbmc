#include <cassert>
#include <type_traits>

struct Plain
{
  int a;
  int b;
};
struct WithVirt
{
  virtual ~WithVirt();
  int a;
};
struct NoCopy
{
  NoCopy(const NoCopy &) = delete;
  NoCopy &operator=(const NoCopy &) = delete;
  NoCopy();
};

int main()
{
  static_assert(std::is_standard_layout<Plain>::value, "");
  static_assert(!std::is_standard_layout<WithVirt>::value, "");
  static_assert(std::is_trivial<Plain>::value, "");
  static_assert(!std::is_trivial<NoCopy>::value, "");
  static_assert(std::is_aggregate_v<Plain>, "");
  static_assert(std::is_assignable<int &, int>::value, "");
  static_assert(!std::is_assignable<int, int>::value, "");
  static_assert(std::is_copy_assignable<Plain>::value, "");
  static_assert(!std::is_copy_assignable<NoCopy>::value, "");
  static_assert(std::is_copy_constructible<Plain>::value, "");
  static_assert(!std::is_copy_constructible<NoCopy>::value, "");
  static_assert(std::is_destructible<Plain>::value, "");
  static_assert(!std::is_destructible<void>::value, "");
  static_assert(std::is_nothrow_move_constructible<Plain>::value, "");
  static_assert(std::is_nothrow_move_assignable<Plain>::value, "");
  static_assert(std::conjunction_v<std::true_type, std::true_type>, "");
  static_assert(!std::conjunction_v<std::true_type, std::false_type>, "");
  static_assert(std::disjunction_v<std::false_type, std::true_type>, "");
  static_assert(!std::disjunction_v<std::false_type, std::false_type>, "");
  static_assert(std::negation_v<std::false_type>, "");

  static_assert(std::is_same<std::remove_cvref_t<const int &>, int>::value, "");
  static_assert(
    std::is_same<std::remove_cvref_t<volatile int &&>, int>::value, "");
  // Only top-level cv is stripped: a pointer to const keeps its pointee cv.
  static_assert(
    !std::is_same<std::remove_cvref_t<const int *>, int *>::value, "");

  std::aligned_storage_t<sizeof(Plain), alignof(Plain)> buf;
  assert(sizeof(buf) >= sizeof(Plain));
  return 0;
}
