#include <stdlib.h>

int main(int argc, char **argv) {
  int *a = (int *)malloc(sizeof(int));

  int *temp = realloc(a, 2 * sizeof(int));
  a = temp;

  free(a);
  return 0;
}
