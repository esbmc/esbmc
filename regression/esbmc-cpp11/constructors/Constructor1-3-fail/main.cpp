#include <cassert>

class t2
{
public:
  int i;

  t2() : i(1)
  {
  }
};

int main()
{
  t2 instance2;
  assert(instance2.i == 0);
}
