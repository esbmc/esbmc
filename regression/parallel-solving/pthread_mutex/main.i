#include <pthread.h>
#include <stdlib.h>

int iRThreads;
void *a;

int main(int argc, char *argv[]) 
{ 
    int i, err; 
    if (argc != 1) 
        a = malloc(sizeof(int)); 
    if (0 != (err = pthread_mutex_init((pthread_mutex_t*)malloc(sizeof(pthread_mutex_t)), NULL)))
        for (;i < iRThreads;) 
            ;
    return 0;
}
