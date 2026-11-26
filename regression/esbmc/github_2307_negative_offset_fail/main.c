#include <string.h>

int main(void)
{
  int a[4] = {1, 2, 3, 4};
  memcpy(a - 1, a + 1, sizeof(int));
  return 0;
}
