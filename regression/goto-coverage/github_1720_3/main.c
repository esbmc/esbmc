#include <assert.h>
#include <stdbool.h>

int main()
{
  int a = 1;
  int b = 2;
  if ((a == 1 && b == 3) || (a != b && true))
  {
    assert(1);
  }
  else
    assert(0);
}