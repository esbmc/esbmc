// #9782: case 1 skips j's initializer, so j is indeterminate there
#include <assert.h>
int nondet_int(void);
int main(void)
{
  switch (nondet_int())
  {
  case 0:
  {
    unsigned char j = 3;
  case 1:
    assert(j >= 3);
  }
  }
  return 0;
}
