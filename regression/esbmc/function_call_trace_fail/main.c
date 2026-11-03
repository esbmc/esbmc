#include <assert.h>

void leaf(int value)
{
  assert(value == 8);
}

int main(void)
{
  leaf(7);
  return 0;
}
