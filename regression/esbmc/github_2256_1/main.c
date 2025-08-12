#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct node {
  int data;
  struct node *nextPtr;
} nodet;

int main() {
  // allocates memory
  nodet *node1 = (nodet *)malloc(sizeof(nodet));
  node1->data = 15;

  // print data
  assert(node1->data == 15);
  printf("node1: %i\n", node1->data);

  // Deallocates memory allocated by malloc
  free(node1);
  return 0;
}

