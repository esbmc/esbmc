// Issue #5137: the data-race flag was indexed by a flat hash
// `pointer_object * 1000 + pointer_offset`, so a write to a large array element
// (offset >= 1000) spilled into another object's flag band and fabricated a
// data race. Here `data` is written by exactly one thread and `arr[k]` by
// another at a solver-chosen offset; the program is race-free and must verify.
#include <pthread.h>

extern unsigned __VERIFIER_nondet_uint(void);

int data;
int arr[4096];

void *write_arr(void *p)
{
  unsigned k = __VERIFIER_nondet_uint();
  if (k < 4096)
    arr[k] = 1;
  return 0;
}

void *write_data(void *p)
{
  data = 2;
  return 0;
}

int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, write_arr, 0);
  pthread_create(&b, 0, write_data, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
