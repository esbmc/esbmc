#include <time.h>
#include <assert.h>

int main()
{
  time_t t = time(NULL);
  assert(t >= 0);
  return 0;
}
