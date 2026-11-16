// #9375: map::insert(first, last) must not overwrite a key already present.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  std::map<int, int> o;
  o[1] = 99;
  o[2] = 20;
  m.insert(o.begin(), o.end());
  assert(m.size() == 2);
  assert(m[1] == 99);
  assert(m[2] == 20);
  return 0;
}
