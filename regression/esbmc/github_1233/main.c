#include <assert.h>

void inner(int x)
{
  assert(x != 2);
}

void middle(int x)
{
  inner(x + 1);
}

int main()
{
  middle(1);
  return 0;
}
