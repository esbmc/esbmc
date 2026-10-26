#include <cassert>

struct C
{
  int *p;
  C(int v) : p(new int(v))
  {
  }
  C(const C &o) : p(new int(*o.p))
  {
  }
  ~C()
  {
    delete p;
  }
};

int main()
{
  try
  {
    throw C(5);
  }
  catch (const C &c)
  {
    assert(*c.p != 5);
  }
  return 0;
}
