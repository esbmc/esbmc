#include <cassert>

int ii = 1;

class t2
{
public:
  int i;

  t2() : i(2)
  {
  }

  ~t2() { ii = 0; }
};

int main()
{
  t2 *p = new t2;
  assert(p->i == 2);
  delete p;
  assert(ii == 0);
}
