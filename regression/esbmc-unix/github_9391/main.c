#include <assert.h>
#include <unistd.h>

/* #9391: getopt() returns the option character it finds, so "prog -a"
 * reaches case 'a' and violates the assertion. */
int main(int argc, char **argv)
{
  int aflag = 0, c;
  while ((c = getopt(argc, argv, "a")) != -1)
    switch (c)
    {
    case 'a':
      aflag = 1;
      break;
    default:
      return 1;
    }
  assert(aflag == 0);
  return 0;
}
