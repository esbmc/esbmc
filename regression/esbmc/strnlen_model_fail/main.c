#include <string.h>

int main()
{
  char buf[4] = {'a', 'b', 'c', 'd'};
  return strnlen(buf, 5);
}
