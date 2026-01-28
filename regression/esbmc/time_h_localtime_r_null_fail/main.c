#include <time.h>

int main()
{
  struct tm buf;
  struct tm *result = localtime_r((void *)0, &buf);
  (void)result;
  return 0;
}
