
#include <stddef.h>
#include <assert.h>

typedef struct {
	int i, j, k;
} pthread_mutex_t;

typedef struct {
	long int silly;
} pthread_t;

#define PTHREAD_MUTEX_INITIALIZER { 0, 0, 0 }

static volatile int cnt = 0;

int pthread_mutex_lock(pthread_mutex_t *);
int pthread_mutex_unlock(pthread_mutex_t *);
int pthread_create(pthread_t *, void *, void *(*)(void *), void *);
int pthread_join(pthread_t, void **);

void *f(void *data)
{
	pthread_mutex_t *mtx = data;
	pthread_mutex_lock(mtx);
	cnt++;
	pthread_mutex_unlock(mtx);
}

int main()
{
	// pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_t mtx;
	pthread_mutex_init(&mtx, NULL);
	int n = 2;
	pthread_t th[n];
	for (int i=0; i<n; i++)
		pthread_create(th+i, NULL, f, &mtx);
	for (int i=0; i<n; i++)
		pthread_join(th[i], NULL);
	assert(cnt == n);
}
