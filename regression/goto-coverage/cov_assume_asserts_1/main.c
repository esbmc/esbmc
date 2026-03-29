#include <assert.h>

void func(int x)
{
  assert(x != 1);
  if (x > 10)
  {
    x = x + 1;
  }
}

int main()
{
  int x = 1;
  func(x);
}
