// Flexible array member + offsetof container pattern
#include <stdlib.h>
#include <stddef.h> // offsetof


struct node {
int id;
size_t len;
char data[];
};


void use_id(int *p) { *p; }


int main() {
/* allocate node with space for 8 bytes of data */
struct node *n = malloc(sizeof *n + 8);
n->id = 7;
n->len = 8;


/* get pointer to the flexible array member */
char *pdata = n->data;


/* recover container using offsetof */
void *tmp = ((void*)pdata) - offsetof(struct node, data);
struct node *now = (struct node *) tmp;


use_id(&(now->id));
free(n);
return 0;
}

