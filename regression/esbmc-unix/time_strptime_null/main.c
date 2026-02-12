#include <time.h>

int main()
{
  struct tm tm;
  /* NULL format should trigger assertion failure */
  char *ret = strptime("2024-01-15", (char *)0, &tm);
  (void)ret;
  return 0;
}
