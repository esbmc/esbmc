// #9895: %s reads a buffer whose lifetime has ended.
#include <stdio.h>

int main()
{
  char *data;
  {
    char buf[4] = "abc";
    data = buf;
  }
  printf("%s\n", data);
  return 0;
}
