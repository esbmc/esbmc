#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[3] = 30;
  m[5] = 50;

  // Miss: the first key not less than / greater than the argument.
  assert(m.lower_bound(2)->first == 3);
  assert(m.upper_bound(2)->first == 3);
  assert(m.lower_bound(0)->first == 1);
  assert(m.upper_bound(0)->first == 1);

  // Hit: lower_bound stops on the key, upper_bound steps past it.
  assert(m.lower_bound(3)->first == 3);
  assert(m.upper_bound(3)->first == 5);

  // Past the last key.
  assert(m.lower_bound(9) == m.end());
  assert(m.upper_bound(9) == m.end());

  assert(m.lower_bound(2)->second == 30);
  return 0;
}
