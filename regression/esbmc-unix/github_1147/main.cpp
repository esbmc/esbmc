#include <pthread.h>

pthread_t __ESBMC_get_thread_id(void);

int main(){
    int threadid;
    threadid = __ESBMC_get_thread_id();
    
    return 0;
}
