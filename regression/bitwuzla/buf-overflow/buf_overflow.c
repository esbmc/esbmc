#include <stdio.h>
#include <string.h>

char buffer[10];

int main(int argc, char **argv)
{
  int a = 10;
  for(int i = 0; i < a + 1; i++)
  {
    printf("ptr[%d] = '%c'\n", i, buffer[i]);
  }
  return 0;
}
