#include <assert.h>

void test()
{
  int x = 1;
  assert(x == 1);
}

int test1()
{
  return 2;
}

int main()
{
  test();
  int y = test1();
  assert(y++ > 0);
}