// #10749: the registration sits on the ')' closing the call, past the
// parentheses inside the string argument and the (void) cast.
#include <assert.h>
#include <pthread.h>
void *t(void *arg)
{
  assert(*(const char *)arg != 'a');
  return 0;
}
int main(void)
{
  pthread_t id;
  (void)pthread_create(&id, 0, t, (void *)"a)\"(");
  pthread_join(id, 0);
  return 0;
}
