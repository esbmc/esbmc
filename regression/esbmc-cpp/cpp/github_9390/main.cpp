// #9390: queue(const Container&) and stack(const Container&) initialise the
// adaptor with the container's elements ([queue.cons], [stack.cons]).
#include <cassert>
#include <deque>
#include <list>
#include <queue>
#include <stack>

int main()
{
  std::deque<int> d{1, 2};
  std::queue<int> q(d);
  assert(!q.empty() && q.size() == 2 && q.front() == 1 && q.back() == 2);
  q.pop();
  assert(q.front() == 2);

  std::list<int> l{1, 2};
  std::queue<int, std::list<int>> ql(l);
  assert(ql.size() == 2 && ql.front() == 1 && ql.back() == 2);

  std::stack<int, std::list<int>> s(l);
  assert(s.size() == 2 && s.top() == 2);
  s.pop();
  assert(s.top() == 1);
  return 0;
}
