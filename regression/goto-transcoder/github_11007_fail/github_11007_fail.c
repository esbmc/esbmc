// #11007: an overflowing fortified copy must still be reported.
#include <string.h>
int main()
{
  char a[5] = {1, 2, 3, 4, 5}, b[4];
  __builtin___memcpy_chk(b, a, 5, __builtin_object_size(b, 0));
  return 0;
}
