#include<stdlib.h>
#include<stdio.h>

struct list_el {
   struct list_el * next;
};

typedef struct list_el item;


int main(void) {
  item *curr, *head, *ret;
  int i;

  head = NULL;

  curr = (item *)malloc(sizeof(item));
  curr->next  = head;

  while(curr) {
    curr = curr->next;
  }

	return 0;
}
