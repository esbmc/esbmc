#include <stdio.h>
#include <stdlib.h>

int main(void) {
  char *test = malloc(20);
  if (test == NULL) { return 1; }
  for (int i = 0; i < 20; ++i) {
    test[i] = 'a' + i;
  }
  test[19] = '\0';
  free(test);
  printf("test = %s\n", test);
  return 0;
}
