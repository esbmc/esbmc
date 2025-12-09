#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main()
{
  char *s = (char *)malloc(100);
  strncpy(s, "runoob", 6);
  int x = printf("%s", s);
}