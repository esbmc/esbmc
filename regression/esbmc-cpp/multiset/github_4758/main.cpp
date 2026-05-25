// Regression for issue #4758: std::multiset::reverse_iterator copy ctor must
// take a const reference, otherwise copying the prvalue returned by
// rbegin()/rend() fails to parse under --std c++14.
#include <set>
#include <cassert>

int main()
{
  std::multiset<int> s;
  for (int i = 0; i < 5; ++i)
    s.insert(i);

  int highest = *s.rbegin();
  assert(highest == 4);

  return 0;
}
