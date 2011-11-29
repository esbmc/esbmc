#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <errno.h>
#include <pthread.h>
#include <sys/types.h>


typedef struct bounded_buf_tag
{
  int valid;

  pthread_mutex_t mutex;
  pthread_cond_t  not_full;
  pthread_cond_t  not_empty;
  
  size_t   item_num;
  size_t   max_size;
  size_t   head, rear;

  size_t   p_wait;  // waiting producers
  size_t   c_wait;  // waiting consumers

  void **  buf;
}bounded_buf_t;

#define BOUNDED_BUF_VALID 0xACDEFA

#define BOUNDED_BUF_INITIALIZER \
   { BOUNDED_BUF_VALID,  PTHREAD_MUTEX_INITIALIZER, \
     PTHREAD_COND_INITIALIZER,  PTHREAD_COND_INITIALIZER, \
     0, 0, 0, 0, null }


int bounded_buf_init(bounded_buf_t * bbuf, size_t sz)
{
  int status = 0; 
  
  if (bbuf == NULL) return EINVAL;
  
  bbuf->valid = BOUNDED_BUF_VALID;

  pthread_mutex_init(&bbuf->mutex, NULL);
  pthread_cond_init(&bbuf->not_full, NULL);

  pthread_cond_init(&bbuf->not_empty, NULL);
  if (status != 0)
  {
    pthread_cond_destroy(&bbuf->not_full);
    pthread_mutex_destroy(&bbuf->mutex);
    return status;
  }

  bbuf->item_num = 0;
  bbuf->max_size = sz;
  bbuf->head = 0;
  bbuf->rear = 0;
  bbuf->buf = malloc( sz * sizeof(void*) );
  if (bbuf->buf == NULL)
  {
    pthread_mutex_destroy(&bbuf->mutex);
    pthread_cond_destroy(&bbuf->not_full);
    pthread_cond_destroy(&bbuf->not_empty);
    return ENOMEM;
  }

  memset(bbuf->buf, 0, sizeof(void*) * sz );  
  bbuf->head = bbuf->rear = 0;
  return 0;
}

bounded_buf_t buffer;

int main(int argc, char **argv)
{

  int i; 

  bounded_buf_init(&buffer, 3);

  assert(0);

  return 0;
}
