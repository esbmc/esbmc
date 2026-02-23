#include <time.h>
#include <assert.h>

int main()
{
  time_t a = time(NULL);
  time_t b = time(NULL);
  double d = difftime(a, b);
  assert(d == 0.0);
  return 0;
}
