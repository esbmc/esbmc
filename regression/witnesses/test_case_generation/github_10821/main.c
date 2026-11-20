#include <assert.h>
extern unsigned char __VERIFIER_nondet_uchar(void);

int main(void)
{
  unsigned char a[3];
  for (int i = 0; i < 3; i++)
    a[i] = __VERIFIER_nondet_uchar();
  assert(a[2] != 7);
  return 0;
}
