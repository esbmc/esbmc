/* #10905: a pointer made from an object's address keeps that object, so
 * moving it below the object is still out of bounds */
#include <stdint.h>
int main(void)
{
  char a[4] = {0};
  char *p = (char *)(uintptr_t)a - 1;
  return *p;
}
