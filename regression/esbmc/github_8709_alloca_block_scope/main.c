#include <stddef.h>
#include <assert.h>
void *alloca(size_t size);
int main(void)
{
  int *p;
  {
    p = alloca(sizeof(int));
    *p = 1;
  }
  assert(*p == 1);
  return 0;
}
