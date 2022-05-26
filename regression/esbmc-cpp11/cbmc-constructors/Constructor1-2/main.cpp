#include <cassert>
class t1
{
public:
  int i;

  t1()
  {
    i = 1;
  }
};

int main()
{
  t1 instance1;
  assert(instance1.i == 1);
}
