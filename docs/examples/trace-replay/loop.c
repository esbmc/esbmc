#include <assert.h>

int main(void)
{
  int sum = 0;
  for (int i = 0; i < 3; ++i)
  {
    sum += i;
  }
  assert(sum == 4);
  return 0;
}
