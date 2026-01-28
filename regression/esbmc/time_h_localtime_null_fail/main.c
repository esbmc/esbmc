#include <time.h>

int main()
{
  struct tm *result = localtime((void *)0);
  (void)result;
  return 0;
}
