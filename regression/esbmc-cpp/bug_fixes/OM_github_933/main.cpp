#include <cassert>

typedef struct
{
  int __lock;
  unsigned int __futex;
  unsigned int __nwaiters;
} pthread_cond_t;

int main()
{
  assert(1);
  return 0;
}
