#include <cassert>

int order = 0;

struct G
{
  int id;
  G(int i) : id(i)
  {
  }
  ~G()
  {
    order = order * 10 + id;
    assert(order != 21);
  }
};

G a(1);
G b(2);

int main()
{
  return 0;
}
