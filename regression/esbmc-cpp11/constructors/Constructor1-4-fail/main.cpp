#include <cassert>
class t3
{
public:
  int i;

  t3();
};

t3::t3() : i(1)
{
}

int main()
{
  t3 instance3;
  assert(instance3.i == 0);
}
