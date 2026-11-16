#include <cassert>
#include <cstdlib>

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
      assert(order == 4321);
  }
};

struct Automatic
{
  ~Automatic()
  {
    assert(0);
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

void handler()
{
  order = order * 10 + 4;
}

void leave()
{
  Automatic automatic;
  exit(*o.p - 1);
}

int main()
{
  constructed();
  atexit(handler);
  leave();
  assert(0);
  return 0;
}
