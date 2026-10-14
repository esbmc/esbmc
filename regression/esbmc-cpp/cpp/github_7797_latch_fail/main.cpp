// #7797: a latch that has not reached zero must not report ready.
#include <latch>
#include <cassert>

int main()
{
  std::latch gate(2);
  gate.count_down();
  assert(gate.try_wait());
  return 0;
}
