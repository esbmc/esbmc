#include <assert.h>

void leaf(int value)
{
  assert(value == 7);
}

int main(void)
{
  leaf(7);
  return 0;
}
