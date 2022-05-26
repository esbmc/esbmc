#include <cassert>
class t1
{
public:
  int i;

  t1()
  {
    assert(0);
  }
};

int main()
{
  t1 instance1;
}
