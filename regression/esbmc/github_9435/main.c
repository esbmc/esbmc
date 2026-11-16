#include <assert.h>
#include <errno.h>
#include <semaphore.h>
#include <time.h>

sem_t s, t;

int main(void)
{
  sem_init(&s, 0, 1);
  assert(sem_trywait(&s) == 0);
  errno = 0;
  assert(sem_trywait(&s) == -1 && errno == EAGAIN);
  sem_post(&s);
  assert(sem_trywait(&s) == 0);

  struct timespec ts = {0, 0};
  sem_init(&t, 0, 1);
  assert(sem_timedwait(&t, &ts) == 0);
  return 0;
}
