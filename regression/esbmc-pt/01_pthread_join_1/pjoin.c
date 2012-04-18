#include <pthread.h>

int aglobal = 0;

void *
thread1(void *dummy)
{

	__ESBMC_atomic_begin();
	aglobal++;
	__ESBMC_atomic_end();
	return NULL;
}

void *
thread2(void *dummy)
{

	__ESBMC_atomic_begin();
	aglobal++;
	__ESBMC_atomic_end();
	pthread_exit(NULL);
}

int
main()
{
	pthread_t p1, p2;

	pthread_create(&p1, NULL, thread1, NULL);
	pthread_create(&p2, NULL, thread2, NULL);

	// Use pthread_join as synchronisation.
	pthread_join(p1, NULL);
	pthread_join(p2, NULL);

	// If the threads are still running, this can/will fail.
	assert(aglobal == 2);
	return 0;
}
