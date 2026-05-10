#include <cassert>

namespace std
{
struct strong_ordering
{
  signed char __value_;
  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};
inline constexpr strong_ordering strong_ordering::less{-1};
inline constexpr strong_ordering strong_ordering::equal{0};
inline constexpr strong_ordering strong_ordering::greater{1};
} // namespace std

// Pin the same-base pointer fold in cmp_three_way2t::do_simplify.
// (&arr[i] <=> &arr[j]) folds at simplify-time when i, j are constants,
// so the SMT layer never sees the ITE chain.
int main()
{
  int arr[5] = {0, 1, 2, 3, 4};

  assert(((&arr[1]) <=> (&arr[3])).__value_ < 0);
  assert(((&arr[2]) <=> (&arr[2])).__value_ == 0);
  assert(((&arr[4]) <=> (&arr[0])).__value_ > 0);

  return 0;
}
