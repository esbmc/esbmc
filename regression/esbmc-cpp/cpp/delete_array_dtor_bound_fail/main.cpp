#include <cassert>

int dtors = 0;

struct E
{
  int a;
  virtual ~E()
  {
    dtors++;
  }
};

int main()
{
  E *p = new E[3];
  delete[] p;
  assert(dtors == 4);
}
