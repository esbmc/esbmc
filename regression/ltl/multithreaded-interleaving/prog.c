#include <pthread.h>

int s, a;

void *w1(void *arg)
{
	a = 1;
	return 0;
}

void *w2(void *arg)
{
	if (a)
		s = 1;
	return 0;
}

int main()
{
	pthread_t t1, t2;
	s = 0;
	pthread_create(&t1, 0, w1, 0);
	pthread_create(&t2, 0, w2, 0);
	pthread_join(t1, 0);
	pthread_join(t2, 0);
	return 0;
}
