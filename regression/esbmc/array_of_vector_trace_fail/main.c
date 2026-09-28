#include <assert.h>
typedef int v4i __attribute__((__vector_size__(16)));
int main(void)
{
  v4i a[3] = {};
  for (int j = 0; j < 4; j++)
    a[2][j] = 20 + j;
  v4i x = a[2];
  assert(x[0] == 21);
  return 0;
}
