// #9357: a reference to an element survives the erase of another one.
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  m[3] = 30;
  int &r = m[2];
  m.erase(1);
  assert(r != 20);
  return 0;
}
