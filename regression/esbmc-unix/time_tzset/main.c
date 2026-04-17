#include <time.h>
#include <assert.h>

int main()
{
  /* Call tzset to initialize timezone variables */
  tzset();

  /* daylight should be 0 or 1 */
  assert(daylight == 0 || daylight == 1);

  return 0;
}
