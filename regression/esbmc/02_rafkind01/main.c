#include <pthread.h>
//#include <assert.h>

int x;

void foo( pthread_mutex_t * m ){
        pthread_mutex_lock( m );
        x += 1;
        pthread_mutex_unlock( m );
}

void * xx( void * arg ){
        pthread_mutex_t * mutex = (pthread_mutex_t *) arg;
        foo( mutex );
        return NULL;
}

int main(){
        pthread_t t;
        pthread_mutex_t mutex;
        pthread_mutex_init( &mutex, NULL );
        pthread_create( &t, NULL, xx, &mutex );
        pthread_join( t, NULL );
}
