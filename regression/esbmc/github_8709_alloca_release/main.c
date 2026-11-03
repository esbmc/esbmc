#include <stddef.h>
void *alloca(size_t size);
void f(void)
{
  int *p = alloca(sizeof(int));
  *p = 1;
}
int main(void)
{
  f();
  return 0;
}
