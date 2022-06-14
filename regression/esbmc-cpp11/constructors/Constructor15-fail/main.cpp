#include <cassert>
class t2
{
public:
  int a[5];

  t2() : a{0,1,2,3,4}
  {
  }
};

int main()
{
  t2 t;
  assert(t.a[0] != 0);
}
