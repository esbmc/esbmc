#include <pthread.h>

int main() 
{ 
    for (;;) 
        pthread_join(0, 0);
}
