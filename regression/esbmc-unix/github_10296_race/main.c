#include <pthread.h>
int x = 0;
void *t(void *a) { x = 1; return 0; }
int main() { pthread_t a; pthread_create(&a, 0, t, 0); x = 2; pthread_join(a,0); return 0; }
