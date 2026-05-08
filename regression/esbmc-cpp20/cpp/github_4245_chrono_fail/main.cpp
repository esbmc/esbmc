// github.com/esbmc/esbmc/issues/4245 — wrong cross-period count comparison.
#include <chrono>
#include <cassert>

int main()
{
  // microseconds(2000) and milliseconds(2) represent the same time, but
  // their raw counts (2000 vs 2) must differ — this assertion is false.
  assert(
    std::chrono::microseconds(2000).count() ==
    std::chrono::milliseconds(2).count());
  return 0;
}
