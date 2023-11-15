#include <stdio.h>
#include <assert.h>

int main()
{
  char *s = "abcde1234151";
  int data = 1000000;
  int x = printf("%s%d\n", s, data);
#ifdef _WIN32
  assert(x == 20);
#else
  assert(0); // fails in linux/macos
#endif
}