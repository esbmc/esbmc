#include <stddef.h>
void *alloca(size_t size);
int *f(void)
{
  int *p = alloca(sizeof(int));
  *p = 1;
  return p;
}
int main(void)
{
  int *p = f();
  return *p;
}
