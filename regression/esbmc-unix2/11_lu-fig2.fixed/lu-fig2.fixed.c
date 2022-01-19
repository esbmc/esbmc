#include <pthread.h> 
#include <stdlib.h>
#include <assert.h>

int mThread=0;
int start_main=0;
int mStartLock=0;
int __COUNT__ =0;

void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}

void acquire(void) {
    __VERIFIER_atomic_begin();
    assume_abort_if_not(mStartLock == 0);
    mStartLock = 1;
    __VERIFIER_atomic_end();
}
void release() {
    __VERIFIER_atomic_begin();
    assume_abort_if_not(mStartLock == 1);
    mStartLock = 0;
    __VERIFIER_atomic_end();
}

int thr1() { //nsThread::Init (mozilla/xpcom/threads/nsThread.cpp 1.31)

  int PR_CreateThread__RES = 1;
  acquire();
  start_main=1;
  { __VERIFIER_atomic_begin();
      if( __COUNT__ == 0 ) { // atomic check(0);
	mThread = PR_CreateThread__RES; 
	__COUNT__ = __COUNT__ + 1; 
    __VERIFIER_atomic_end();
      } else { assert(0); } 
  }
  release();
  if (mThread == 0) { return -1; }
  else { return 0; }

}

void thr2() { //nsThread::Main (mozilla/xpcom/threads/nsThread.cpp 1.31)

  int self = mThread;
  while (start_main==0);
  acquire();
  release();
  { __VERIFIER_atomic_begin();
      if( __COUNT__ == 1 ) { // atomic check(1);
	    int rv = self; // self->RegisterThreadSelf();
	    __COUNT__ = __COUNT__ + 1;
      } else { assert(0); } 
    __VERIFIER_atomic_end();
  }
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}

