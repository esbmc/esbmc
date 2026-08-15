// github.com/esbmc/esbmc/issues/6984 — the clocks are nondeterministic, so a
// reading is not pinned to the epoch and two readings need not differ.
// Both assertions must be reachable failures, or the clock model is a
// constant dressed up as a clock.
#include <chrono>
#include <cassert>

using namespace std::chrono;

int main()
{
  auto a = steady_clock::now();
  assert(a.time_since_epoch().count() == 0);

  auto b = steady_clock::now();
  assert(b > a);

  return 0;
}
