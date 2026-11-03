// The same anchor, one level deeper.
#include <assert.h>
unsigned char p[2][3][4];
int main(void)
{
  int *e = (int *)&p[0][0][0];
  assert((char *)e == (char *)p[0][0]);
  return 0;
}
