#include <assert.h>

int main()
{
  int x = 2147483647ULL;
  assert((x + 1) == 2147483648ULL);
}
