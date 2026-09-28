#include <stdio.h>
#include <stdlib.h>

int main()
{
  int year;
  scanf("%d", &year);
  if (year < 1900 || year > 3000)
    return 0;
  int *age = malloc(sizeof(int));
  if (!age)
    return 0;
  *age = year - 1900;
  printf("Age: %d\n", *age);
  free(age);
  return 0;
}
