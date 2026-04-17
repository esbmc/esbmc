#include <time.h>
#include <assert.h>

int main()
{
  time_t stored;
  time_t returned = time(&stored);
  assert(returned == stored);
  return 0;
}
