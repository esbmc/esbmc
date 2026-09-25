// github.com/esbmc/esbmc/issues/6984 — <chrono> time_point, time_point_cast
// and the three clocks.
#include <chrono>
#include <cassert>
#include <ctime>

using namespace std::chrono;

int main()
{
  time_point<system_clock> tp;
  assert(tp.time_since_epoch().count() == 0);

  system_clock::time_point epoch;
  auto later = epoch + seconds(5);
  assert(duration_cast<seconds>(later.time_since_epoch()).count() == 5);
  assert(later > epoch);
  assert(later - epoch == seconds(5));
  assert(seconds(5) + epoch == later);
  assert(later - seconds(5) == epoch);

  auto t = epoch;
  t += seconds(3);
  assert(t - epoch == seconds(3));
  t -= seconds(1);
  assert(t - epoch == seconds(2));

  assert(time_point_cast<seconds>(later).time_since_epoch().count() == 5);
  assert(system_clock::to_time_t(later) == (std::time_t)5);
  assert(system_clock::from_time_t(7) == epoch + seconds(7));

  // [time.clock.steady]: readings never go backwards.
  auto a = steady_clock::now();
  auto b = steady_clock::now();
  assert(b >= a);
  assert((b - a).count() >= 0);

  auto h = high_resolution_clock::now();
  assert(h >= b);

  static_assert(steady_clock::is_steady, "steady_clock is steady");
  static_assert(!system_clock::is_steady, "system_clock is not steady");

  return 0;
}
