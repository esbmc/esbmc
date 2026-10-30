#include <string.h>

int main(void)
{
  char name[4] = "abc";
  return memchr(name, 'z', 64) != NULL;
}
