#include <pthread.h>

/* lock1: 运行时初始化 -> 触发 pthread_mutex_init (389) */
pthread_mutex_t lock1;
/* lock2: 静态初始化 -> 首次加锁时触发 pthread_mutex_initializer (402) */
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

int counter = 0;

void *worker(void *arg)
{
  /* pthread_mutex_lock -> _check / _nocheck / _noassert (456 / 425 / 414) */
  pthread_mutex_lock(&lock1);
  counter++;
  /* pthread_mutex_unlock -> _check / _nocheck / _noassert (490 / 445 / 436) */
  pthread_mutex_unlock(&lock1);

  /* pthread_mutex_trylock (501)，并触发 lock2 的 initializer (402) */
  if (pthread_mutex_trylock(&lock2) == 0)
  {
    counter++;
    pthread_mutex_unlock(&lock2);
  }
  return 0;
}

int main()
{
  pthread_t t1, t2;

  pthread_mutex_init(&lock1, 0); /* pthread_mutex_init (389) */

  pthread_create(&t1, 0, worker, 0);
  pthread_create(&t2, 0, worker, 0);

  pthread_join(t1, 0);
  pthread_join(t2, 0);

  /* pthread_mutex_destroy (540)；--lock-order-check 下为 destroy_check (521) */
  pthread_mutex_destroy(&lock1);
  pthread_mutex_destroy(&lock2);

  return 0;
}
