// Nested structs and array members (kernel-style hlist) to exercise offsetof in macros
#include <stdlib.h>
#include <stddef.h> // offsetof
#define typeof __typeof__


struct hlist_node { struct hlist_node *next; };


struct container {
int val;
struct hlist_node link;
};


#define container_of(ptr, type, member) ({ \
void *__mptr = (void *)(ptr); \
((type *)(__mptr - offsetof(type, member))); })


void touch(int *p) { *p; }


int main() {
struct container *c = malloc(sizeof *c);
c->val = 99;
c->link.next = NULL;


struct hlist_node *first = &c->link;
struct container *recov = container_of(first, struct container, link);
touch(&(recov->val));


free(c);
return 0;
}
