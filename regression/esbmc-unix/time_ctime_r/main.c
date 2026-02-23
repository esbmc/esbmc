#include <time.h>
#include <assert.h>

int main()
{
  time_t t = 1705322400; /* Some timestamp */
  char buf[26];
  char *ret = ctime_r(&t, buf);

  /* ctime_r should return the buffer on success */
  assert(ret == buf);
  /* Buffer should be null-terminated */
  assert(buf[25] == '\0');

  return 0;
}
