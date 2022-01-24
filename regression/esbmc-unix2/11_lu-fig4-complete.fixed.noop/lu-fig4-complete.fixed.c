#include "lu-fig4.h"
#include <pthread.h>

bool __START_ASYNC__ = False; // models if the second thread can start
int __COUNT__ = 0; // models a counter which monitors order violations

inline OSErr PBReadSync(int foo) { return noErr; }

inline void DoneWaitingOnThisThread(PRThread thread)
{
    int is;
    int thread_md_asyncIOLock;
    int thread_io_pending;
    int thread_md_asyncIOCVar;

	_PR_INTSOFF(is);
	PR_Lock(thread_md_asyncIOLock);
	{ __ESBMC_atomic_begin();
	if (__COUNT__ == 1) {
	  thread_io_pending = PR_FALSE; // check for order violation
	  __COUNT__ = __COUNT__ + 1;
	} else {
	  assert(0);
	}
     __ESBMC_atomic_end();
	}
	// let the waiting thread know that async IO completed
	PR_NotifyCondVar(thread_md_asyncIOCVar);
	PR_Unlock(thread_md_asyncIOLock);
	_PR_FAST_INTSON(is);
}

// ====================== 1st thread
PRInt32 readWriteProc(PRFileDesc fd, void* buf, PRUint32 bytes, IOOperation op)
{
	assert(0);
	return -1;
}

// ====================== 2nd thread

static void* asyncIOCompletion (ExtendedParamBlock pbAsyncPtr_thread)
{
  while (__START_ASYNC__ == False) {}; // second thread waits until the async read is issued
    PRThread thread = pbAsyncPtr_thread;

    if (_PR_MD_GET_INTSOFF()) {
		return NULL;
    }
    _PR_MD_SET_INTSOFF(1);

	DoneWaitingOnThisThread(thread);

    _PR_MD_SET_INTSOFF(0);

}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, readWriteProc, NULL);
  pthread_create(&t2, NULL, asyncIOCompletion, NULL);
  return 0;
}


inline bool _PR_MD_GET_INTSOFF() { return PR_FALSE; }
/* 
retractall(preds(_,_,_)), retractall(trans_preds(_,_,_,_)),
assert(preds(1, p(_,data(__CNT__,__START_ASYNC__,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)), [__CNT__=0,__CNT__>=1,__START_ASYNC__=<0])),
assert(preds(2, p(_,data(__CNT__,__START_ASYNC__,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)), [__START_ASYNC__>=0,__START_ASYNC__=<0,__START_ASYNC__>=1,__CNT__>=1,__CNT__=<1])),
assert(trans_preds(_-1, p(_,data(__CNT__,__START_ASYNC__,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)), p(_,data(C,D,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)), [__CNT__-C=0,__START_ASYNC__>=1,D=<0])),
assert(trans_preds(_-2, p(_,data(__CNT__,__START_ASYNC__,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)), p(_,data(C,D,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)), [__START_ASYNC__-D=<0,__START_ASYNC__-D>=0,__CNT__-C=0,C>=1,C=<1])).
*/
