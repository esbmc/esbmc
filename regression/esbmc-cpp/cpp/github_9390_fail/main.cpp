// #9390: a queue built from a two-element deque is not empty.
#include <cassert>
#include <deque>
#include <queue>

int main()
{
  std::deque<int> d{1, 2};
  std::queue<int> q(d);
  assert(q.empty());
  return 0;
}
