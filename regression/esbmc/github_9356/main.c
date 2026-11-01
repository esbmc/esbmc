#include <string.h>
#include <time.h>

/* #9356: asctime returns a 25-character string, which overflows buf. */
int main(void)
{
  struct tm t = {0};
  t.tm_year = 100;
  t.tm_mday = 1;
  t.tm_wday = 6;
  char buf[16];
  strcpy(buf, asctime(&t));
  return buf[0];
}
