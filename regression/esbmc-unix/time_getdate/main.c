#include <time.h>
#include <assert.h>

int main()
{
  struct tm *result = getdate("January 1, 2024");

  /* getdate returns NULL on error, valid pointer on success */
  if (result != (struct tm *)0)
  {
    /* On success, tm fields should be in valid ranges */
    assert(result->tm_mon >= 0 && result->tm_mon <= 11);
    assert(result->tm_mday >= 1 && result->tm_mday <= 31);
  }
  else
  {
    /* On error, getdate_err should be set (1-8) */
    assert(getdate_err >= 1 && getdate_err <= 8);
  }

  return 0;
}
