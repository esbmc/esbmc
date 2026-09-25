#include <atomic>
#include <cassert>

int main()
{
  std::atomic<int> a{5};
  int expected = 9;
  // CAS with a wrong expected value fails and writes back the current
  // value; asserting success must be violated.
  assert(a.compare_exchange_strong(expected, 11));
  return 0;
}
