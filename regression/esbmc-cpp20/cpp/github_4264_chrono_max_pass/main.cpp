// github.com/esbmc/esbmc/issues/4264 — duration::max/min/zero static members.
#include <chrono>
#include <cstdint>
#include <climits>
#include <cassert>

using Usecs = std::chrono::duration<int64_t, std::micro>;

static int64_t take(Usecs t = Usecs::max())
{
  return t.count();
}

int main()
{
  assert(Usecs::max().count() == LLONG_MAX);
  assert(Usecs::min().count() == LLONG_MIN);
  assert(Usecs::zero().count() == 0);

  // Default-argument site — the original failure mode in #4264.
  assert(take() == LLONG_MAX);
  assert(take(Usecs(42)) == 42);
  return 0;
}
