#include <stdlib.h>
#include <pthread.h>

// This is based on the weaver benchmark

int lA=2, lB=2;
int *A, *B; // will be initialized with NONDET
int iA, iB; // initialized with: 0


// increments iA until A[iA] != B[iA]
void* thread1() {
  //int c = 0;
  while(iA < lA && iA < lB)
    if(A[iA] == B[iA]) iA++;     
    else break;

  return 0;
}

// increments iB until A[iB] != B[iB]
void* thread2() {

  while(iB < lB && iB < lA)
    if(A[iB] == B[iB]) iB++;
    else break;

  return 0;
}


int main() {
  // Nondet arrays
  A = malloc(sizeof(int) * lA);
  B = malloc(sizeof(int) * lB);
  
  pthread_t t1, t2;
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  
  __ESBMC_assert(iA == iB, "iA and iB must be equal");
}
