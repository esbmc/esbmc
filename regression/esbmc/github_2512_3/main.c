// offsetof on a union member (union as first member)
#include <stdlib.h>
#include <stddef.h>


struct S {
union {
int a;
char c;
} u;
int tail;
};


void use_tail(int *p) { *p; }


int main() {
struct S *s = malloc(sizeof *s);
s->u.a = 5;
s->tail = 13;


/* pointer to union member */
void *pu = &s->u;


void *tmp = ((void*)pu) - offsetof(struct S, u);
struct S *now = (struct S *) tmp;


use_tail(&(now->tail));
free(s);
return 0;
}

