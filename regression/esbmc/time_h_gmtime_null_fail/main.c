#include <time.h>

int main()
{
  struct tm *result = gmtime((void *)0);
  (void)result;
  return 0;
}
