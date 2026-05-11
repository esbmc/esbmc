#include <cassert>

// In-source comparison-category type — exercises the IREP2
// cmp_three_way2t lowering without depending on the external <compare>
// OM.  All three category types share a signed-char first field, which
// is what the SMT layer writes when it expands cmp_three_way to the
// less / equivalent / greater ITE chain.
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

int main()
{
  std::strong_ordering lt = 1 <=> 2;
  assert(lt.__value_ < 0);

  std::strong_ordering eq = 5 <=> 5;
  assert(eq.__value_ == 0);

  std::strong_ordering gt = 9 <=> 3;
  assert(gt.__value_ > 0);

  // Pointer spaceship is also strong_ordering for ordered pointers.
  int arr[3] = {0, 1, 2};
  int *p = &arr[0];
  int *q = &arr[2];
  std::strong_ordering po = p <=> q;
  assert(po.__value_ < 0);

  return 0;
}
