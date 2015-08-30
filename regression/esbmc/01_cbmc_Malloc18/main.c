#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

struct list_el {
   int val;
   struct list_el *next;
};

typedef struct list_el item;

item *curr, *head, *ret;

item* search_list(item* i, int k) {
  int x;
  while(i!=NULL && i->val!=k) {
    i = i->next ;
  }
  return i;
}

void insert_list(item* i, int k) {
	if(i==NULL){
		i=(item*)malloc(sizeof(item));
                __ESBMC_assume(i);
		i->val=k;
		i->next=NULL;
	}else{
		i=(item *)malloc(sizeof(item));
                __ESBMC_assume(i);
		i->val=k;
		i->next=NULL;
		head=i;
	}
}

int main(void) {
  int i=0;
  head = NULL;

  for(i=1;i<=2;i++) {
    curr = (item *)malloc(sizeof(item));
    __ESBMC_assume(curr);
    curr->val = i;
    curr->next  = head;
    head = curr;
  }

  insert_list(curr,3);
  ret = search_list(curr,3);
  assert(ret->val==3);

  return 0;
}

