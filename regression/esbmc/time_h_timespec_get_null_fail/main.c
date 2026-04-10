#include <time.h>

int main()
{
  int result = timespec_get((void *)0, TIME_UTC);
  (void)result;
  return 0;
}
