#include <assert.h>
#include <stddef.h>
void *alloca(size_t size);
int main(void)
{
  void *p;
  for (int i = 0; i < 2; ++i)
  {
    p = alloca(sizeof(int));
    *(int *)p = i;
  }
  alloca(sizeof(int));
  assert(*(int *)p == 1);
  return 0;
}
