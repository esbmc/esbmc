#include <time.h>
#include <assert.h>

int main()
{
  struct tm tm;
  char *ret = strptime("2024-01-15", "%Y-%m-%d", &tm);

  /* strptime returns either NULL or a valid pointer */
  /* The returned tm fields should be in valid ranges if parsed */
  if (ret != (char *)0)
  {
    assert(tm.tm_mon >= 0 && tm.tm_mon <= 11);
    assert(tm.tm_mday >= 1 && tm.tm_mday <= 31);
  }

  return 0;
}
