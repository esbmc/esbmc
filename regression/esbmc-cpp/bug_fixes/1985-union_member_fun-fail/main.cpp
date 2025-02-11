#include <cassert>

union a
{
  int b()
  {
    return 22;
  }
};
int main()
{
  a c;
  assert(c.b() == 22222);
}
