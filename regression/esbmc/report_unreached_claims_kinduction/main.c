#include <assert.h>
void never_called(int v)
{
  assert(v > 0);
}
int main(void)
{
  int i = 0;
  while (i < 3)
    i++;
  assert(i == 3);
  return 0;
}
