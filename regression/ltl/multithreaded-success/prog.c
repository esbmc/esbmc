#include <pthread.h>
int s, other;
void *worker(void *arg) { other = 1; return 0; }
int main()
{
	pthread_t t;
	s = 0;
	pthread_create(&t, 0, worker, 0);
	s = 0;
	pthread_join(t, 0);
	return 0;
}
