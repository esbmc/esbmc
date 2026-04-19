#include <time.h>
#include <assert.h>

int main()
{
  time_t t = 1000000;
  struct tm *result = localtime(&t);
  assert(result != (void *)0);
  return 0;
}
