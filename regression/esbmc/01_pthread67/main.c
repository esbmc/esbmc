/* Extracted from https://github.com/sosy-lab/sv-benchmarks. */

#include <pthread.h>

void __VERIFIER_error(void);
typedef struct _barrier
{
  int thread_count;
  int seen;
  int pass;
  _Bool leaving;
  pthread_mutex_t lock;
  pthread_cond_t let_in, let_out;
} Barrier;
void barrier_init(Barrier *b, int thread_count)
{
  (!(thread_count > 1) ? __VERIFIER_error() : (void)0);
  b->thread_count = thread_count;
  b->seen = 0;
  b->pass = 0;
  b->leaving = 0;
  pthread_mutex_init(&b->lock, ((void *)0));
  pthread_cond_init(&b->let_in, ((void *)0));
  pthread_cond_init(&b->let_out, ((void *)0));
}
void barrier_destroy(Barrier *b)
{
  pthread_mutex_destroy(&b->lock);
  pthread_cond_destroy(&b->let_in);
  pthread_cond_destroy(&b->let_out);
}
_Bool barrier_wait(Barrier *b)
{
  pthread_mutex_lock(&b->lock);
  while(b->leaving)
  {
    pthread_cond_wait(&b->let_in, &b->lock);
  }
  ++b->seen;
  if(b->seen == b->thread_count)
  {
    b->pass = b->thread_count - 1;
    b->leaving = 1;
    b->seen = 0;
    pthread_cond_broadcast(&b->let_out);
    pthread_mutex_unlock(&b->lock);
    return 1;
  }
  while(!b->leaving)
  {
    pthread_cond_wait(&b->let_out, &b->lock);
  }
  if(--b->pass == 0)
  {
    b->leaving = 0;
    pthread_cond_broadcast(&b->let_in);
  }
  pthread_mutex_unlock(&b->lock);
  return 0;
}
typedef struct
{
  Barrier *b1, *b2;
  int tid;
} Arg;
volatile _Bool pre[2], in[2], post[2], sig1[2], sig2[2];
void *worker_fn(void *arg)
{
  Arg *a = arg;
  const int tid = a->tid;
  pre[tid] = 1;
  for(int i = 0; i < 2; ++i)
  {
    (!(!in[i]) ? __VERIFIER_error() : (void)0);
    (!(!post[i]) ? __VERIFIER_error() : (void)0);
  }
  sig1[tid] = barrier_wait(a->b1);
  int sig = 0;
  for(int i = 0; i < 2; ++i)
  {
    (!(pre[i]) ? __VERIFIER_error() : (void)0);
    sig += sig1[i];
  }
  (!(sig <= 1) ? __VERIFIER_error() : (void)0);
  (!(!in[tid]) ? __VERIFIER_error() : (void)0);
  in[tid] = 1;
  sig2[tid] = barrier_wait(a->b2);
  (!(!post[tid]) ? __VERIFIER_error() : (void)0);
  post[tid] = 1;
  sig = 0;
  for(int i = 0; i < 2; ++i)
  {
    (!(pre[i]) ? __VERIFIER_error() : (void)0);
    (!(in[i]) ? __VERIFIER_error() : (void)0);
    sig += sig1[i];
  }
  (!(sig == 1) ? __VERIFIER_error() : (void)0);
  sig = 0;
  for(int i = 0; i < 2; ++i)
  {
    sig += sig2[i];
  }
  (!(sig <= 1) ? __VERIFIER_error() : (void)0);
  return ((void *)0);
}
int main()
{
  Barrier b1, b2;
  Arg a[2];
  for(int i = 0; i < 2; ++i)
  {
    a[i].b1 = &b1;
    a[i].b2 = &b1;
    a[i].tid = i;
  }
  barrier_init(&b1, 2);
  pthread_t worker[2 - 1];
  for(int i = 0; i < 2 - 1; ++i)
    pthread_create(&worker[i], ((void *)0), &worker_fn, &a[i]);
  worker_fn(&a[2 - 1]);
  for(int i = 0; i < 2 - 1; ++i)
    pthread_join(worker[i], ((void *)0));
  int sig = 0;
  for(int i = 0; i < 2; ++i)
  {
    (!(post[i]) ? __VERIFIER_error() : (void)0);
    sig += sig2[i];
  }
  (!(sig == 1) ? __VERIFIER_error() : (void)0);
}
