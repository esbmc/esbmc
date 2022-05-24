#include <assert.h>
#include <string.h>
int main()
{
  int z;
  int *p = &(int){0};
  memcpy(p, &z, sizeof(z));
  assert(*p == z);
}
