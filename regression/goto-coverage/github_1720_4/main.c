#include <assert.h>
#include <stdbool.h>

int main()
{
  int a = 1;
  int b = 2;
  while(a && (!b && !a || !b) && true)
  {
    assert(1);
  }
}