#include <stdio.h>
#include <string.h>
//#include <cheri/cheric.h>
#include "cheric.h"

char *buffer = "hello";
char *secret = "secret";

int main(int argc, char **argv)
{
  char *__capability cap_ptr = cheri_ptr(buffer, 9);
  /* Overflow buffer, leaking the secret in a traditional system */
  for(int i = 0; i < strlen(buffer); i++)
  {
    printf("ptr[%d] = '%c'\n", i, cap_ptr[i]);
  }
  return 0;
}
