extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }

extern int __VERIFIER_nondet_int();
/*
 * Simple example: Build a list of the form 1->2->3... (arbitrary length, at least 10 values) 
 * Afterwards, go through it and check if the list does have the value 2 and 6;
 * 
 * List implementation is modified from BLAST benchmarks.
 *
 * This source code is licensed under the GPLv3 license.
 */
#include <stdlib.h>

void myexit(int s) {
	_EXIT: goto _EXIT;
}

typedef struct node {
  int h;
  struct node *n;
} *List;

int main() {
  /* Build a list of the form 1->2->3->4... */
  List a = (List) malloc(sizeof(struct node));
  if (a == 0) myexit(1);
  List t;
  List p = a;

  int counter = 0;
  while (counter < 10 || __VERIFIER_nondet_int()) {
    p->h = counter;
    t = (List) malloc(sizeof(struct node));
    if (t == 0) myexit(1);
    p->n = t;
    p = p->n;
    counter++;
  }
  p->h = counter;
  p->n = 0;
  p = a;

  int hasTwo = 0;
  int hasTwelve = 0;

  while (p!=0) {
    if (p->h == 2) {
      hasTwo = 1;
    }

    if (p->h == 12) {
      hasTwelve = 1;
    }
    p = p->n;
  }

  if(!hasTwelve || !hasTwo) {
    ERROR: {reach_error();abort();}
  }

  return 0;
}

