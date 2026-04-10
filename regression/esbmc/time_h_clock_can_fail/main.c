#include <time.h>
#include <assert.h>

int main()
{
  clock_t c = clock();
  assert(c != (clock_t)(-1));
  return 0;
}
