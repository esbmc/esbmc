#include <stdio.h>
#include <assert.h>

int main()
{
  char *s = "abcde1234151";
  int data = 1000000;
  int x = printf("%s%d\n", s, data);
  // In WINDOWS
  // error log: "no function body for __stdio_common_vfprintf"
  // therefore symex_prinf is not executed, leading to 'x = 0'
  // this may be fixed by adding sth like (base_name == "__stdio_common_vfprintf")
  assert(x == 20);
  x += 1;
}