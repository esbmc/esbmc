#include <cassert>

int ii = 1;

class t2
{
public:
  int i;
  int j;

  t2(int _i, int _j) : i(_i), j(_j)
  {
  }

  ~t2() { ii = 0; }
};

int main()
{
  t2 *p = new t2(2, 3);
  assert(p->i == 2);
  assert(p->j == 3);
  delete p;
  assert(ii == 1);
}
