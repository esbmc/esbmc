#include <cassert>

class C
{
  int buf[4];

public:
  unsigned n;
  C() : n(0)
  {
  }
  void push(int v)
  {
    buf[n++] = v;
  }
};

int main()
{
  C c;
  for (int i = 0; i < 5; i++)
    c.push(i);
  assert(c.n <= 4);
  return 0;
}
