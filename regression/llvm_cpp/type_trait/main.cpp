#include<assert.h>

int main(void)
{
  if(__is_pod(int))
    return 0;

  assert(0);
  return 1;
}
