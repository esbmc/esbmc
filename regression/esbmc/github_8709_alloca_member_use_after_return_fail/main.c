#include <stddef.h>
void *alloca(size_t size);
struct s
{
  void *m;
};
void *f(void)
{
  struct s v;
  v.m = alloca(sizeof(int));
  *(int *)v.m = 1;
  return v.m;
}
int main(void)
{
  int *p = f();
  return *p;
}
