/* #10905: a pointer made from an object's address keeps that object, so
 * moving it past the object is still out of bounds */
#include <stdint.h>
int main(void)
{
  int a[2] = {0};
  int *p = (int *)(uintptr_t)a + 3;
  *p = 1;
  return 0;
}
