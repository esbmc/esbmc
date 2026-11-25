#include <string.h>
int main()
{
  char b[2];
  memccpy(b, "abcd", 'z', 4);
  return 0;
}
