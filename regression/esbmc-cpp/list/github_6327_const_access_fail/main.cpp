// Negative companion to github_6327_const_access: front() of the const list is
// 1, so asserting a wrong value is violated.
#include <cassert>
#include <list>

int main()
{
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  const std::list<int> &cl = l;
  assert(cl.front() == 42); // wrong on purpose: front is 1
  return 0;
}
