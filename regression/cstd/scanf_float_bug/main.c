#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
  float *arr = (float *)malloc(3 * sizeof(float));
  for(int i = 0; i < 3; i++)
  {
    scanf("%13f", &arr[i]);
  }
  for(int i = 0; i < 3; i++)
  {
    printf("%f", &arr[i]);
  }
}