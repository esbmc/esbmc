#include <string.h>

extern int __VERIFIER_nondet_int(void);

int main(void)
{
  char d[4];
  int i = __VERIFIER_nondet_int();
  while (i < 0)
  {
    if (i != -5)
      i = i + 1;
  }
  memcpy(d, 0, 0);
  return 0;
}
