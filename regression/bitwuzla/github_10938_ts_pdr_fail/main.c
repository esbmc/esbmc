#include <assert.h>
int nondet_int(void);
char shadow[1 << 20];
int main(void)
{
  int i = nondet_int();
  if (i >= 0 && i < (1 << 20))
    shadow[i] = 1;
  assert(shadow[0] == 0);
  return 0;
}
