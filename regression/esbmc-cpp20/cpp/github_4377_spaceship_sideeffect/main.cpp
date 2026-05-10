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

int counter = 0;
int next()
{
  return ++counter;
}

int main()
{
  // Each operand of <=> must be evaluated exactly once.  A converter-
  // side if/else lowering would call next() in both the < arm and the
  // == arm, producing counter == 4.  The IREP2 cmp_three_way node
  // captures each side in a single SSA name, so counter == 2.
  std::strong_ordering o = next() <=> next();
  assert(o.__value_ < 0);
  assert(counter == 2);
  return 0;
}
