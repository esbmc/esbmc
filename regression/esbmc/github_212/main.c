extern void __VERIFIER_error() __attribute__ ((__noreturn__));

#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

/*
This source code is licensed under the GPLv3 license. 

Author: Alexander Driemeyer.
*/

struct data_struct {
  int number;
  int *array;
};

typedef struct data_struct *Data;

struct node_t {
  Data data;
  struct node_t *next;
};

static Data create_data() {

  // Create optional data

  if(__VERIFIER_nondet_int()) {
    return NULL;
  }

  Data data = malloc(sizeof *data);

  if(__VERIFIER_nondet_int()) {
    data->array = (int*) malloc(20 * sizeof(data->array));

    int counter = 0;

    for(counter = 0; counter < 20; counter++)  {
      data->array[counter] = __VERIFIER_nondet_int();
    }

  } else {
    data->array = NULL;
  }

  data->number = __VERIFIER_nondet_int();

  return data;
}

static void freeData(Data data) {

  if(data == NULL) {
    return;
  }

  if(data->array != NULL) {
    free(data->array);
  }

  free(data);
}

static void append(struct node_t **pointerToList) {
  struct node_t *node = malloc(sizeof *node);
  node->next = *pointerToList;
  node->data = create_data();
  *pointerToList = node;
}

int main() {
  struct node_t *list = NULL;

  /* Create a long singly-linked list with optional data.
  */

  int dataNotFinished = 0;

  do {
    append(&list);
  } while(__VERIFIER_nondet_int());

/*
Do something with data.
  displayData();
*/

//  free list and data
  while (list) {
    struct node_t *next = list->next;
    freeData(list->data);
    free(list);
    list = next;
  }

  return 0;
}
