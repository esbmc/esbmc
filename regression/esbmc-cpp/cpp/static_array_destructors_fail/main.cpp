#include <cassert>

struct G
{
  int id = 0;
  ~G()
  {
    assert(id != 1);
  }
};

struct L
{
  int id = 0;
  ~L()
  {
    assert(id != 2);
  }
};

G g[2];

void f()
{
  static L s[2];
  s[1].id = 2;
}

int main()
{
  g[1].id = 1;
  f();
  return 0;
}
