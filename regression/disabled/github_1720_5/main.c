#include <assert.h>
#include <stdbool.h>

int main()
{
  int a = 1;
  int b = 2;
  if (a == 1 || a == 2 || (b != 2 && (b == 3 || a == 1)))
  {
    assert(1);
  }
}