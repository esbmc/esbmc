#include <cassert>

int order = 0;

struct D
{
  int id = 0;
  ~D()
  {
    order = order * 10 + id;
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

struct Holder
{
  Owner owner;
};

struct Check
{
  ~Check()
  {
    assert(order == 654321);
  }
};

Check check;
D g[2][2];
Owner owned[2];
Holder held[2];

void constructed()
{
  static D s[2];
  s[0].id = 5;
  s[1].id = 6;
}

void never_called()
{
  static D n[2];
  n[0].id = 9;
}

int main()
{
  g[0][0].id = 1;
  g[0][1].id = 2;
  g[1][0].id = 3;
  g[1][1].id = 4;
  constructed();
  return *owned[1].p - 1;
}
