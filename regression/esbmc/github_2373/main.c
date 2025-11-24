#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int num, ind, newsize, i = 1;

void *expandArray(int **pointer) {
  while (newsize < num) {
    newsize = newsize + 1;
    int *temp = realloc(*pointer, sizeof(int) * newsize);
    if (temp != NULL) {
      temp[newsize - 1] = i;
      *pointer = temp;
    } else {
      return 0;
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  num = __VERIFIER_nondet_int();
  if (!(num > 0 && num < 100)) {
    return 0;
  }
  int *a = (int *)malloc(sizeof(int));
  if (a == NULL) {
    return 0;
  }
  newsize = 0;
  expandArray(&a);
  free(a);
  return 0;
}
