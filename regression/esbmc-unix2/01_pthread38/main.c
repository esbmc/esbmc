#include <pthread.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void *funcA(void *param) {
  assert(0);
}

int main(void) {
  int err;
  pthread_t id;

  if (0 != (err = pthread_create(&id, NULL, &funcA, NULL))) {
    fprintf(stderr, "Error [%d] found creating 2stage thread.\n", err);
    exit(-1);
  }

  return 0;
}
