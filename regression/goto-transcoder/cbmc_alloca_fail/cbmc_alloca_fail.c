#include <alloca.h>
int main()
{
  int *p = (int *)alloca(4 * sizeof(int));
  p[7] = 9; // out of bounds
  return 0;
}
