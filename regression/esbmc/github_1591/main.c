#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
  srand(time(0));
  for (int i = 0; i < 100; i++)
  {
    printf("%d\n", rand());
  }
  int a, b, n;

  a = rand() + 1;
  b = rand() + 1;

  n = a / b;
  printf("%d\n", n);
}
