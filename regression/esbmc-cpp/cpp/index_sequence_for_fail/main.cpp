#include <cassert>
#include <utility>

int main()
{
  std::index_sequence_for<int, char, long> s;

  // The sequence has one index per type, so its size is 3.
  assert(s.size() == 2);

  return 0;
}
