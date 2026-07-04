#include <alloca.h>
int main()
{
  int *p = (int *)alloca(4 * sizeof(int));
  p[0] = 5;
  p[3] = 9;
  return 0;
}
