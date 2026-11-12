#include <string.h>

int main(void)
{
  char a[2] = {'a', 'b'};
  int r = strcoll(a, "ab");
  char out[4];
  strxfrm(out, "hello", 6);
  return r;
}
