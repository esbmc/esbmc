#include <cassert>
class t2
{
public:
  int i;

  t2() : i(2)
  {
  }
};

int main()
{
  t2 *p = new t2;
  assert(p->i == 2);
  delete p;
}
