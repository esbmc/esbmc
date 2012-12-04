#include <pthread.h> 
int idx; // bit idx = 0; controls which of the two elements ctr1 or ctr2 will be used by readers
int ctr1, ctr2; // byte ctr[2];
int readerprogress1, readerprogress2, readerprogress3; // byte readerprogress[N_QRCU_READERS];
int mutex; // bit mutex = 0; updates are done in critical section, only one writer at a time
int NONDET;

void* qrcu_reader1() {
  int myidx;
  
  /* rcu_read_lock */
  while (1) {
    myidx = idx;
    if (NONDET) {
      { __ESBMC_atomic_begin();
	__ESBMC_assume(myidx <= 0);
	__ESBMC_assume(ctr1>0);
	ctr1++;
        __ESBMC_atomic_end();
      }
      break;
    } else {
      if (NONDET) {
	{ __ESBMC_atomic_begin();
	  __ESBMC_assume(myidx > 0);
	  __ESBMC_assume(ctr2>0);
	  ctr2++;
      __ESBMC_atomic_end();
	}
	break;
      } else {}
    }
  }

  readerprogress1 = 1; /*** readerprogress[me] = 1; ***/
  readerprogress1 = 2; /*** readerprogress[me] = 2 ***/

  /* rcu_read_unlock */
  { __ESBMC_atomic_begin();
      if (myidx <= 0) { ctr1--; } // use ctr1
      else { ctr2--; } // use ctr2
   __ESBMC_atomic_end();
  }
}

/* sums the pair of counters forcing weak memory ordering */
#define sum_unordered \
  if (NONDET) {       \
    sum = ctr1;       \
    sum = sum + ctr2; \
  } else {            \
    sum = ctr2;       \
    sum = sum + ctr1; \
  }

void* qrcu_updater() {
  int i;
  int readerstart1;
  int readerstart2;
  int readerstart3;
  int sum;

  glb_init(idx==0);
  glb_init(ctr1==1);
  glb_init(ctr2==0);
  glb_init(readerprogress1==0);
  glb_init(readerprogress2==0);
  glb_init(readerprogress3==0);
  glb_init(mutex==0);  

  /* Snapshot reader state. */
  { __ESBMC_atomic_begin();
      readerstart1 = readerprogress1;
      readerstart2 = readerprogress2;
      readerstart3 = readerprogress3;
    __ESBMC_atomic_end();
  }

  sum_unordered;
  if (sum <= 1) { sum_unordered; }
  if (sum > 1) {
    acquire(mutex);
    if (idx <= 0) { ctr2++; idx = 1; ctr1--; }
    else { ctr1++; idx = 0; ctr2--; }
    if (idx <= 0) { while (ctr2 > 0); }
    else { while (ctr1 > 0); }
    release(mutex);
  }

  /* Verify reader progress. */
  { __ESBMC_atomic_begin();
      if (NONDET) {
	__ESBMC_assume(readerstart1 == 1);
	__ESBMC_assume(readerprogress1 == 1);
	assert(0);
      } else {
	if (NONDET) {
	  __ESBMC_assume(readerstart2 == 1);
	  __ESBMC_assume(readerprogress2 == 1);
	  assert(0);
	} else {
	  if (NONDET) {
	    __ESBMC_assume(readerstart2 == 1);
	    __ESBMC_assume(readerprogress2 == 1);
	    assert(0);
	  } else { }
	}
      }
   __ESBMC_atomic_end();
  }
}

void* qrcu_reader2() {
  int myidx;
  
  /* rcu_read_lock */
  while (1) {
    myidx = idx;
    if (NONDET) {
      { __blockattribute__((atomic))
	__ESBMC_assume(myidx <= 0);
	__ESBMC_assume(ctr1>0);
	ctr1++;
      }
      break;
    } else {
      if (NONDET) {
	{ __blockattribute__((atomic))
	  __ESBMC_assume(myidx > 0);
	  __ESBMC_assume(ctr2>0);
	  ctr2++;
	}
	break;
      } else {}
    }
  }

  readerprogress2 = 1; /*** readerprogress[me] = 1; ***/
  readerprogress2 = 2; /*** readerprogress[me] = 2 ***/

  /* rcu_read_unlock */
  { __blockattribute__((atomic))
      if (myidx <= 0) { ctr1--; } // use ctr1
      else { ctr2--; } // use ctr2
  }
}

void* qrcu_reader3() {
  int myidx;
  
  /* rcu_read_lock */
  while (1) {
    myidx = idx;
    if (NONDET) {
      { __blockattribute__((atomic))
	__ESBMC_assume(myidx <= 0);
	__ESBMC_assume(ctr1>0);
	ctr1++;
      }
      break;
    } else {
      if (NONDET) {
	{ __blockattribute__((atomic))
	  __ESBMC_assume(myidx > 0);
	  __ESBMC_assume(ctr2>0);
	  ctr2++;
	}
	break;
      } else {}
    }
  }

  readerprogress3 = 1; /*** readerprogress[me] = 1; ***/
  readerprogress3 = 2; /*** readerprogress[me] = 2 ***/

  /* rcu_read_unlock */
  { __blockattribute__((atomic))
      if (myidx <= 0) { ctr1--; } // use ctr1
      else { ctr2--; } // use ctr2
  }
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, qrcu_reader1, NULL);
  pthread_create(&t2, NULL, qrcu_updater, NULL);
  return 0;
}

#define acquire_thread_id(tid, l) \
  { __blockattribute__((atomic)) \
    __ESBMC_assume(l==0); \
    l = tid; \
  }

