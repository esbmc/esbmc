#include <pthread.h>

int s;

void *worker(void *arg)
{
	s = 1;
	return 0;
}

int main()
{
	pthread_t t;
	s = 0;
	pthread_create(&t, 0, worker, 0);
	pthread_join(t, 0);
	s = 0;
	return 0;
}
