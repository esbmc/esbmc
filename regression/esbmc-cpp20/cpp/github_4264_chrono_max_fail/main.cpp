// github.com/esbmc/esbmc/issues/4264 — Usecs::max() must NOT be zero.
#include <chrono>
#include <cstdint>
#include <cassert>

int main()
{
  using Usecs = std::chrono::duration<int64_t, std::micro>;
  assert(Usecs::max().count() == 0); // false — should be LLONG_MAX
  return 0;
}
