#include <pthread.h>
#include <assert.h>

int g = 0;
int other = 0;

struct S
{
  int *p;
  int *p2;
};

struct S s = {&other, &g};

// R29, prefix siblings: "p" is a prefix of "p2". Selecting the shorter name
// resolves this write to `other`, the race against g disappears and the
// interleaving is pruned -- so this reports SUCCESSFUL unless the longest
// declared name wins.
void *writer(void *arg)
{
  (void)arg;
  *(s.p2) = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
