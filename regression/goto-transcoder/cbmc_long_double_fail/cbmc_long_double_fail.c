#include <assert.h>
int main()
{
  long double d = 1.5L;
  assert(d == 2.5L); // false: confirms the value is really decoded
  return 0;
}
