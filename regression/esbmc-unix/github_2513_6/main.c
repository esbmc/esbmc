extern int __VERIFIER_nondet_int();
extern void abort(void);
#include <assert.h>
void reach_error() { }

#include <stdlib.h>

typedef struct node {
    int val;
    struct node *next;
} Node;

int main() {
    Node *p, *list = malloc(sizeof(*list));
    Node *tail = list;
    list->next = NULL;
    list->val = 10;
    while (__VERIFIER_nondet_int()) {
        int x = __VERIFIER_nondet_int();
        if (x < 10 || x > 20) continue;
        p = malloc(sizeof(*p));
        tail->next = p;
        p->next = NULL;
        p->val = x;
        tail = p;
    }

    while (1) {
        for (p = list; p!= NULL; p = p->next) {
            if (!(p->val <= 20 && p->val >= 10))
                {reach_error();}
            if (p->val < 20) p->val++;
            else p->val /= 2;
        }
    }
}

