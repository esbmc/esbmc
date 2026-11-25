/* #10296's program with a join, which orders the write before the assertion. */
#include <pthread.h>
#include <assert.h>
int x = 0;
void *t(void *a) { x = 5; return 0; }
int main() { pthread_t a; pthread_create(&a, 0, t, 0); pthread_join(a, 0); assert(x == 5); return 0; }
