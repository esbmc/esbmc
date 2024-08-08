#include <cassert>

class Base
{
public:
  int ss[1];
  int a;
  int sss[1];
  Base(): ss{1}, a{2}, sss{3} {}
};

int main()
{
  Base x;
  assert(x.ss[0] == 1);
  assert(x.a == 2);
  assert(x.sss[0] == 3);
}
