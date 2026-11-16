#include <assert.h>

int main()
{
  int i;
  for (i = 0; i < 5; i++)
    ;
  assert(i == 5);
  return 0;
}
