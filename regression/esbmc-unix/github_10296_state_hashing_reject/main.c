/* #10296: --schedule --state-hashing crashed in generate_hash. */
#include <pthread.h>
#include <assert.h>
int x = 0;
void *t(void *a) { x = 5; return 0; }
int main() { pthread_t a; pthread_create(&a, 0, t, 0); assert(x == 5); return 0; }
