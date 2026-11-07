#include <assert.h>
union U
{
  unsigned int x;
  unsigned char c[4];
};
union U a[2] = {{1}, {2}};
int main(void)
{
  unsigned long long v = *(unsigned long long *)a;
  assert((unsigned int)v != 1);
  return 0;
}
