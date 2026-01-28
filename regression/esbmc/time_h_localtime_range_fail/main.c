#include <time.h>
#include <assert.h>

int main()
{
  time_t t = time(NULL);
  struct tm *r = localtime(&t);
  if (r != (void *)0)
  {
    assert(r->tm_sec == 0);
  }
  return 0;
}
