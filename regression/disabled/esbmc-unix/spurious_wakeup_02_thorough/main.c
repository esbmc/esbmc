/* Contributed by: Vladimír Štill, https://divine.fi.muni.cz
    Description: A test of spurious wakeup of pthread_cond_wait.
 */

 #include <pthread.h>
 pthread_mutex_t lock;
 pthread_cond_t cond;

 // only accessed in critical section guarded by mutex, so there is no need to
 // make this variable atomic or volatile
 int x;

 void *thread( void *arg ) {
     (void)arg;
     pthread_mutex_lock( &lock );
     // BUG: cond.wait can be waken up spuriously (see man pthread_cond_wait)
     pthread_cond_wait( &cond, &lock );
     assert( x <= 42 );
     pthread_mutex_unlock( &lock );
     return NULL;
 }

 int main() {
     pthread_t t;
     pthread_mutex_init(&lock,0);
     pthread_cond_init(&cond,0);
     pthread_create( &t, NULL, thread, NULL );
     for ( int i = 0; i <= 42; i++ )
         x = i;
     pthread_cond_broadcast( &cond );
     pthread_join( t, NULL );
 }

