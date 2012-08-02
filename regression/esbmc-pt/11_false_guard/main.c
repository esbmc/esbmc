#include <pthread.h>
#include <stdbool.h>

int nondet_int(void);
int global = 0;
int badgers;
int batsignal = 0;

void *
thread(void *dummy)
{
  assert(batsignal == 0);
}

int
main()
{
  badgers = nondet_int();
  int nou;
  pthread_t face;
  pthread_create(&face, NULL, thread, NULL);

  // Plase a false guard into state guard
  if (global == 1) {
    nou++;
  }

  // And now when removing it, we remove a non-false guard, so the false sticks
  // around.
  if (badgers == 0) {
    nou++;
  }

  // Enable assertion in the other thread; it doesn't fire because false is
  // still in the thread guard.
  batsignal = 1;

  return 0;
}
