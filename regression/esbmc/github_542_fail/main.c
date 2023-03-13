#include <stdlib.h>

int **g = NULL;

void free_g1()
{
  free(g);
  g = NULL;
}

void free_g2()
{
  if(g != NULL)
    free(*g);
}

void f()
{
  *g = (int *)malloc(sizeof(int));
  atexit(free_g1);
  exit(1);
}

int main()
{
  g = (int **)malloc(sizeof(int *));
  atexit(free_g2);
  f();
  return 0;
}
