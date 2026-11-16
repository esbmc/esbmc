#include <assert.h>
#include <unistd.h>

/* #9391: getopt() returns the option character it finds; with at most two
 * options read, each flag is set by its own case. */
int main(int argc, char **argv)
{
  int aflag = 0, bflag = 0, c;
  for (int i = 0; i < 2 && (c = getopt(argc, argv, "ab")) != -1; i++)
    switch (c)
    {
    case 'a':
      aflag = 1;
      break;
    case 'b':
      bflag = 1;
      break;
    default:
      return 1;
    }
  assert(aflag + bflag <= 2);
  return 0;
}
