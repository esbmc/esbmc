#include <assert.h>
/* #10187: prototypes with unnamed parameters of different kinds */
void h(int);
void k(double);
int main()
{
  int x = 1;
  h(2);
  k(1.0);
  assert(x == 2);
  return 0;
}
