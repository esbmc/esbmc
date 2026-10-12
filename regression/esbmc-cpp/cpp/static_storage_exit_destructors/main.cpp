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
    if (id == 1)
      assert(order == 321);
  }
};

struct Owner
{
  int *p;
  Owner() : p(new int(1))
  {
  }
  ~Owner()
  {
    delete p;
  }
};

G a(1);
Owner o;

void constructed()
{
  static G s(3);
}

void never_called()
{
  static G n(9);
}

G b(2);

int main()
{
  constructed();
  constructed();
  assert(order == 0);
  return *o.p - 1;
}
