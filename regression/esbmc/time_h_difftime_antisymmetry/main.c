#include <time.h>
#include <assert.h>

int main()
{
  time_t a = time(NULL);
  time_t b = time(NULL);
  double d1 = difftime(a, b);
  double d2 = difftime(b, a);
  assert(d1 == -d2);
  return 0;
}
