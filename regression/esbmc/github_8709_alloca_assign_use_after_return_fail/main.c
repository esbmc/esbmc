#include <stddef.h>
void *alloca(size_t size);
void *f(void)
{
  void *p;
  p = alloca(sizeof(int));
  *(int *)p = 1;
  return p;
}
int main(void)
{
  int *p = f();
  return *p;
}
