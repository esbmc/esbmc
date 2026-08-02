#include <pthread.h>
#include <semaphore.h>

#define TRUE 1
#define FALSE 0

sem_t prod_done, cons_done;

#define SIZE 2
int data;
int vec_vfy[SIZE];

void* prod_task( void* arg )
{
  int idx;
  for ( idx = 0; idx < SIZE; ++idx )
  {
    data = vec_vfy[idx];
    sem_post( &prod_done );
    sem_wait( &cons_done );
    __ESBMC_assert( 0, "prod: false" );
  }
  __ESBMC_assert( TRUE, "prod_task: done" );
  return NULL;
}

void* cons_task( void* arg )
{
  __ESBMC_assert( TRUE, "cons_task: 1" );
  int idx;
  for ( idx = 0; idx < SIZE; ++idx )
  {
    sem_wait( &prod_done );
    int data1_vfy = vec_vfy[idx];
    int data2_vfy = data;
    __ESBMC_assert( data == vec_vfy[idx], "assert data equal" );
    sem_post( &cons_done );
  }
  __ESBMC_assert( TRUE, "cons_task: done" );
  return NULL;
}

int main()
{
  //------------------------------------------------------------
  sem_init( &prod_done, 0, 0 );
  sem_init( &cons_done, 0, 0 );
  //------------------------------------------------------------
  pthread_t prod_thrd;
  pthread_t cons_thrd;
  //------------------------------------------------------------
  pthread_create( &prod_thrd, NULL, prod_task, NULL );
  pthread_create( &cons_thrd, NULL, cons_task, NULL );
  //------------------------------------------------------------
  pthread_join( prod_thrd, NULL );
  pthread_join( cons_thrd, NULL );
  //------------------------------------------------------------
  sem_destroy( &prod_done );
  sem_destroy( &cons_done );
  //------------------------------------------------------------
}
