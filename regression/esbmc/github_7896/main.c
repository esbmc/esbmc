// #7896: _Float16 is IEEE binary16, whose largest finite value is 65504.
#include <assert.h>

int main(void)
{
  _Float16 x = (_Float16)1000.0f;
  assert(x == 1000.0f);
  return 0;
}
