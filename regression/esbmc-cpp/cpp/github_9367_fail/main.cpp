// #9367: std::unordered_set<int> s{5} holds the single element 5.
#include <cassert>
#include <unordered_set>

int main()
{
  std::unordered_set<int> s{5};
  assert(s.size() != 1);
  return 0;
}
