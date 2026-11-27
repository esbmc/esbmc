#include <cassert>

struct W
{
  int a;
  int b;
};

W F()
{
  return W{1, 2};
}

int main()
{
  W const w{F()};
  assert(w.a == 1);
  assert(w.b == 3);
  return 0;
}
