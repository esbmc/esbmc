#include "lu-fig4.h"
#include <pthread.h> 
bool __START_ASYNC__ = False; // models if the second thread can start
int __COUNT__ = 0; // models a counter which monitors order violations

// ====================== 1st thread
PRInt32 readWriteProc(PRFileDesc fd, void buf, PRUint32 bytes, IOOperation op)
{
        PRInt32 refNum; 
	OSErr err;
	int pbAsync_pb;
	int me_io_pending;

	// quick hack to allow PR_fprintf, etc to work with stderr, stdin, stdout 
	// note, if a user chooses "seek" or the like as an operation in another function 
	// this will not work 
	if (refNum >= 0 && refNum < 3)
	{
		switch (refNum)
		{
				case 0:
				  //stdin - not on a Mac for now
					err = paramErr;
					goto ErrorExit;
					break;
		                case 1: // stdout 
		                case 2: // stderr
					puts();
					break;
		}
		
		return (bytes);
	}
	else
	{
		PRBool  doingAsync = PR_FALSE;

		// 
		// Issue the async read call and wait for the io semaphore associated
		// with this thread.
		// Async file system calls *never* return error values, so ignore their
		// results (see <http://developer.apple.com/technotes/fl/fl_515.html>);
		// the completion routine is always called.
		//
		if (op == READ_ASYNC)
		{
		  //
		  //  Skanky optimization so that reads < 20K are actually done synchronously
		  //  to optimize performance on small reads (e.g. registry reads on startup)
		  //
			if ( bytes > 20480L )
			{
				doingAsync = PR_TRUE;
				{ __ESBMC_atomic_begin();
				if (__COUNT__ == 0) {
				  me_io_pending = PR_TRUE; // check for order violation
				  __COUNT__ = __COUNT__ + 1;
				} else {
				  assert(0);
				}
                 __ESBMC_atomic_end();
				}
				__START_ASYNC__ = True; // second thread can start
				(void)PBReadAsync(pbAsync_pb);
			}
			else
			{
				me_io_pending = PR_FALSE;
				
				err = PBReadSync(pbAsync_pb);
				if (err != noErr && err != eofErr)
					goto ErrorExit;
			}
		}
		else
		{
			doingAsync = PR_TRUE;
			me_io_pending = PR_TRUE;

			// writes are currently always async 
			(void)PBWriteAsync(pbAsync_pb);
		}
		
		if (doingAsync) {
			WaitOnThisThread(PR_INTERVAL_NO_TIMEOUT);
		}
	}
	
	if (err != noErr)
		goto ErrorExit;

	if (err != noErr && err != eofErr)
		goto ErrorExit;
	
	return; 

ErrorExit:
	_MD_SetError(err);
	return -1;
}

inline OSErr PBReadSync(int) { return noErr; }

// ====================== 2nd thread

static void asyncIOCompletion (ExtendedParamBlock pbAsyncPtr_thread)
{
  while (__START_ASYNC__ == False) {}; // second thread waits until the async read is issued
    PRThread thread = pbAsyncPtr_thread;

    if (_PR_MD_GET_INTSOFF()) {
		return;
    }
    _PR_MD_SET_INTSOFF(1);

	DoneWaitingOnThisThread(thread);

    _PR_MD_SET_INTSOFF(0);

}

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
