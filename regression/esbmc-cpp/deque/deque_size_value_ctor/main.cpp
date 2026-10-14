#include <deque>
#include <cassert>

/* deque(size_type, const T&) left _capacity uninitialised before
   verify_capacity() doubled it, so the doubling loop had no bound and
   verification never converged. */
int main()
{
  std::deque<int> d(3, 7);
  assert(d.size() == 3);
  assert(d[0] == 7);
  assert(d[2] == 7);
  return 0;
}
