#include <cassert>

struct a
{
  int b(int i)
  {
    if (i == 0)
    {
      return i;
    }
    else
    {
      return i + b(i - 1);
    }
  }
};
int main()
{
  a a;
  assert(a.b(2) == 3);
}
