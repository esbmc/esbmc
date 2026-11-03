#include <assert.h>
#include <errno.h>
#include <semaphore.h>
#include <time.h>

sem_t s;

int main(void)
{
  struct timespec ts = {0, 0};
  sem_init(&s, 0, 0);
  int r = sem_timedwait(&s, &ts);
  assert(!(r == -1 && errno == ETIMEDOUT));
  return 0;
}
