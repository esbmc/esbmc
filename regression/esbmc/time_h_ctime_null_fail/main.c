#include <time.h>

int main()
{
  char *result = ctime((void *)0);
  (void)result;
  return 0;
}
