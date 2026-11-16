#include <assert.h>
#include <errno.h>
#include <semaphore.h>

sem_t s;

int main(void)
{
  sem_init(&s, 0, 0);
  int r = sem_trywait(&s);
  assert(!(r == -1 && errno == EAGAIN));
  return 0;
}
