// libstdc++ and libc++ make std::min and std::max visible through <ios>,
// <iostream>, <string> and <vector>, without <algorithm> (#7536).
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

struct S
{
  int x;
};

int main()
{
  char storage[sizeof(S)];
  S *s = new (storage) S();
  s->x = 2;
  assert(std::max(1, s->x) == 2);
  assert(std::min(1.5, 2.5) == 1.5);
  return 0;
}
