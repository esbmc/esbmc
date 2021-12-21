#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int getPassword()
{
  char buf[4];
  gets(buf);
  return strcmp(buf, "SMT");
}

int main()
{
  int x = getPassword();
  if(x)
  {
    printf("Access Denied\n");
    exit(0);
  }
  printf("Access Granted\n");
  return 0;
}
