#include <stdio.h>

int main(void)
{
  int scores[5];
  int sum = 0;
  for (int i = 0; i < 5; i++)
    sum += scores[i];
  printf("%d\n", sum);
  return 0;
}
