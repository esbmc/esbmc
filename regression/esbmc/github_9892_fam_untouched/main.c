/* #9892: the flexible array member gives `g` a value-set entry with suffix
 * ".fam[].p", and enumerating the pointers it names crashed. */
#include <stdlib.h>

struct slot
{
  void *p;
};

struct info
{
  void *keep;
  struct slot fam[];
};

struct info g;

int main()
{
  g.keep = malloc(4);
  return 0;
}
