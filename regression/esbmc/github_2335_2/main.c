#include <stdlib.h>
#include <assert.h>

struct nodet {
  struct nodet *n;
  int payload;
};

int main() {
  unsigned i;
  struct nodet *list=(void *)0;
  struct nodet *new_node;
  
  for(i=0; i<2; i++) {
    new_node=malloc(sizeof(*new_node));
    new_node->n=list;
    new_node->payload=i;
    list=new_node;
  }

  assert(new_node->payload==1);
}
