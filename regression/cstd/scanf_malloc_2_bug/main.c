#include <stdio.h>

int main(int argc, char *argv[])
{
  char *toParseStr = (char *)malloc(11);
  printf("Enter string here: ");
  scanf("%12s", toParseStr);
  printf("%s", toParseStr);
  free(toParseStr);
}