// github.com/esbmc/esbmc/issues/6984 — <chrono> duration converting
// constructor and the arithmetic operators.
#include <chrono>
#include <cassert>

using namespace std::chrono;

int main()
{
  // [time.duration.cons] p2: implicit when the period ratio divides exactly.
  nanoseconds ns = seconds(1);
  assert(ns.count() == 1000000000LL);
  milliseconds ms = minutes(2);
  assert(ms.count() == 120000LL);

  // [time.duration.nonmember]: mixed-period arithmetic goes through
  // common_type, so the result carries the finer period.
  auto mixed = seconds(1) + milliseconds(1);
  assert(mixed.count() == 1001);
  assert((seconds(1) - milliseconds(250)).count() == 750);

  assert((seconds(2) * 3).count() == 6);
  assert((3 * seconds(2)).count() == 6);
  assert((seconds(6) / 2).count() == 3);
  assert(seconds(6) / seconds(2) == 3);
  assert((seconds(7) % 4).count() == 3);
  assert((minutes(2) % seconds(50)).count() == 20);

  // [time.duration.arithmetic]: unary and compound forms.
  assert((-seconds(1)).count() == -1);
  assert((+seconds(4)).count() == 4);

  auto d = seconds(2);
  d += seconds(1);
  assert(d.count() == 3);
  d -= seconds(1);
  assert(d.count() == 2);
  d *= 5;
  assert(d.count() == 10);
  d /= 4;
  assert(d.count() == 2);
  d %= seconds(3);
  assert(d.count() == 2);
  assert((++d).count() == 3);
  assert((d--).count() == 3);
  assert(d.count() == 2);

  // Mixed-period comparison.
  assert(seconds(1) == milliseconds(1000));
  assert(milliseconds(1500) > seconds(1));
  assert(microseconds(999) < milliseconds(1));

  // [ratio.ratio] p1: num/den are reduced, and the sign lives on num.
  static_assert(std::ratio<2, 4>::num == 1, "ratio num is reduced");
  static_assert(std::ratio<2, 4>::den == 2, "ratio den is reduced");
  static_assert(std::ratio<1, -2>::num == -1, "ratio sign is on num");
  static_assert(std::ratio<1, -2>::den == 2, "ratio den is positive");

  // Floating-point Rep may truncate, so its converting constructor is
  // unconstrained.
  duration<double> fs = milliseconds(1500);
  assert(fs.count() > 1.4 && fs.count() < 1.6);

  return 0;
}
