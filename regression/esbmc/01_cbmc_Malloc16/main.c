#include<stdlib.h>
#include<stdio.h>

struct list_el {
   int val;
   struct list_el *next;
};

typedef struct list_el item;

item* search_list(item* i, int k) {

  int x;

   while(i!=NULL && i->val!=k) {
      i = i->next ;
   }

	return i;
}

int main(void) {
  item *curr, *head, *ret;
  int i=0;

  head = NULL;

  for(i=1;i<=2;i++) {
    curr = (item *)malloc(sizeof(item));
    __ESBMC_assume(curr);
    curr->val = i;
    curr->next  = head;
    head = curr;
  }

  ret = search_list(curr,2);
  assert(ret->val==2);

  return 0;
}

