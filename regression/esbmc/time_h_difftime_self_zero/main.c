#include <time.h>
#include <assert.h>

int main()
{
  time_t t = time(NULL);
  double d = difftime(t, t);
  assert(d == 0.0);
  return 0;
}
