#include <cassert>

class Base
{
public:
  int a;
  int ss[1];
  int sss[1];
  Base(): a{1}, ss{2}, sss{3} {}
};

int main()
{
  Base x;
  assert(x.a == 1);
  assert(x.ss[0] == 2);
  assert(x.sss[0] == 3);
}
