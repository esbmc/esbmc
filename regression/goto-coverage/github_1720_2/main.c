#include <assert.h>
#include <stdbool.h>

int main()
{
  int a = 1;
  int b = 2;
  if (a == 1 && (b == 2 || a == b) && 1.1)
  {
    assert(1);
  }
  else
    assert(0);
}