extern int __VERIFIER_nondet_int(void);
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
#include <pthread.h>
#define SIZE	(20)
#define EMPTY	(-1)
typedef struct {
    int element[SIZE];
    int head;
    int tail;
    int amount;
} QType;
pthread_mutex_t m;
int __VERIFIER_nondet_int();
int stored_elements[SIZE];
QType queue;
int empty(QType * q) {
  if (q->head == q->tail)  
    return EMPTY;
  else 
    return 0;
}
int enqueue(QType *q, int x) {
  q->element[q->tail] = x;
  q->amount++;
  if (q->tail == SIZE)
    q->tail = 1;
  else 
    q->tail++;
  return 0;
}
void *t1(void *arg) {
  int value, i;
  pthread_mutex_lock(&m);
  value = __VERIFIER_nondet_int();
  if (enqueue(&queue,value)) {
    goto ERROR;
  }
  stored_elements[0]=value;
  if (empty(&queue)) {
    goto ERROR;
  }
  pthread_mutex_unlock(&m);
  return NULL;
  ERROR: __VERIFIER_error();
}
int main(void) {
  pthread_t id1;
  pthread_create(&id1, NULL, t1, &queue);
  return 0;
}
