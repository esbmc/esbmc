#include <stdio.h>
#include <string.h>
#include <cheri/cheric.h>

char *buffer = "hello";
char *secret = "secret";

int main(int argc, char **argv)
{
  char *__capability cap_ptr = cheri_ptr(buffer, 6);
  /* Overflow buffer, leaking the secret in a traditional system */
  for(int i = 0; i < strlen(buffer); i++)
  {
    assert(
      cap_ptr[i] == 'h' || cap_ptr[i] == 'e' || cap_ptr[i] == 'l' ||
      cap_ptr[i] == 'o' || cap_ptr[i] == '\0');
  }
  return 0;
}
