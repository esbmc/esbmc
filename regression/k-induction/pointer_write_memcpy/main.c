// memcpy writes x through a bodyless intrinsic the loop analysis must see.
#include <string.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int x = 0, i = 0;
  for (;;)
  {
    int y = (x + 1) % 10;
    memcpy(&x, &y, sizeof x);
    i++;
    __VERIFIER_assert(x < 10);
  }
}
