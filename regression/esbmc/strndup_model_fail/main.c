#include <string.h>

int main()
{
  char b[3] = {'a', 'b', 'c'};
  strndup(b, 5);
  return 0;
}
