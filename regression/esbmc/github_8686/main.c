/* github #8686: only digits and letters whose value is below the base belong
 * to the subject sequence (C11 7.22.1.4p3). */
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
int main(void)
{
  assert(strtol("19", NULL, 8) == 1);
  assert(strtol("12", NULL, 2) == 1);
  assert(strtol("08", NULL, 0) == 0);
  assert(strtol("z", NULL, 36) == 35);
  assert(strtoll("7fG", NULL, 16) == 0x7f);
  assert(strtoul("Zz", NULL, 36) == 35 * 36 + 35);
  assert(strtoull("0x1F", NULL, 0) == 31);

  char one[2] = "1";
  char *end = one + 1;
  errno = 0;
  assert(strtol(one, &end, 37) == 0 && errno == EINVAL && end == one + 1);
  return 0;
}
