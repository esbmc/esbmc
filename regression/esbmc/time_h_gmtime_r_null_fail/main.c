#include <time.h>

int main()
{
  struct tm buf;
  struct tm *result = gmtime_r((void *)0, &buf);
  (void)result;
  return 0;
}
