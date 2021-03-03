#include <pthread.h>

void *malloc(unsigned size);

typedef struct {
  int key;
  pthread_mutex_t mutex;  
} node;

node *n;

void *thread1(void *arg)
{
  pthread_mutex_lock(&n->mutex); 
  pthread_mutex_unlock(&n->mutex);
}

void *thread2(void *arg)
{
  pthread_mutex_lock(&n->mutex); 
//  pthread_mutex_unlock(&n->mutex);
}

int main(void)
{
  pthread_t id1, id2;

  n = (node *) malloc(sizeof(node));

  pthread_mutex_init(&n->mutex, NULL);
  pthread_create(&id1, NULL, thread1, NULL);
  pthread_create(&id2, NULL, thread2, NULL);

  return 0;
}
