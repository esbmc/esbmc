#include <cassert>
class t1
{
public:
  int i;
};

int main()
{
  t1 instance1;
  instance1.i = 1;
  assert(instance1.i == 1);
}
