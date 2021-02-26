#include <pthread.h>

int a, b, c, d, e;

unsigned int nondet_uint();

void *
thread1(void *dummy)
{
	unsigned int s = nondet_uint();
	__ESBMC_assume(s < 4);
	switch (s) {
		case 0:
			return &a;
		case 1:
			return &b;
		case 2:
			return &c;
		case 3:
			return &d;
	}
}

void *
thread2(void *dummy)
{

	unsigned int s = nondet_uint();
	__ESBMC_assume(s < 4);
	switch (s) {
		case 0:
			pthread_exit(&a);
		case 1:
			pthread_exit(&b);
		case 2:
			pthread_exit(&c);
		case 3:
			pthread_exit(&d);
	}
}

int
main()
{
	pthread_t p1, p2;
	void *exit1, *exit2;

	pthread_create(&p1, NULL, thread1, NULL);
	pthread_create(&p2, NULL, thread2, NULL);

	// Use pthread_join as synchronisation.
	pthread_join(p1, &exit1);
	pthread_join(p2, &exit2);

	// Should pass
	assert(exit1 == &a || exit1 == &b || exit1 == &c || exit1 == &d);
	assert(exit2 == &a || exit2 == &b || exit2 == &c || exit2 == &d);

	// Should fail
	assert(exit1 == &e || exit2 == &e);
	return 0;
}
