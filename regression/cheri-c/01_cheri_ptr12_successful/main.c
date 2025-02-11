#include <stdio.h>
#include <string.h>
#include <cheri/cheric.h>

char *buffer = "hello";
char *secret = "secret";

int main(int argc, char **argv)
{
  // zero-length capability is allowed
  char *__capability cap_ptr = cheri_ptr(buffer, 3);
  assert(cap_ptr[2] == 'l');
  return 0;
}
