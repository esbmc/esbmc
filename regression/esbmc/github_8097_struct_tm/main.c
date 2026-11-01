/* #8097: a struct tm that names its members as glibc does, __tm_zone, differs
 * from the libc model's; the leak check crashed walking the model's localtime
 * buffer. */
#include <stdlib.h>

struct tm
{
  int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday,
    tm_isdst;
  long __tm_gmtoff;
  const char *__tm_zone;
};

struct tm *localtime(const long *);

int *kept;

int main(void)
{
  long t = 0;
  localtime(&t);
  kept = malloc(sizeof(int));
  return 0;
}
