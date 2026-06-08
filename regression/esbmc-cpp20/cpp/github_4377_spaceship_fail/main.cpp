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

int main()
{
  std::strong_ordering gt = 9 <=> 3;
  // gt is greater (.__value_ == 1); the assertion below must fail.
  assert(gt.__value_ < 0);
  return 0;
}
