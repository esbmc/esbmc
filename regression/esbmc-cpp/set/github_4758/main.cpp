// Regression for issue #4758: std::set::reverse_iterator copy ctor must take
// a const reference, otherwise copying the prvalue returned by rbegin()/rend()
// fails to parse under --std c++14 (where mandatory copy elision does not
// hide the missing const).
#include <set>
#include <cassert>

int main()
{
  std::set<int> s;
  for (int i = 0; i < 5; ++i)
    s.insert(i);

  // Forces the copy ctor of reverse_iterator on the prvalue returned by
  // rbegin(): pre-fix this fails with "candidate constructor not viable:
  // expects an lvalue for 1st argument".
  int highest = *s.rbegin();
  assert(highest == 4);

  return 0;
}
