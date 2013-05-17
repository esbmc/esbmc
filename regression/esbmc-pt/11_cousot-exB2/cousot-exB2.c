#include <pthread.h>
int n;
int p1, p2;
int done;

void main() { // the parallel program computes 2^n when n >= 0 (11 transitions)
  int p; 
  glb_init(n >= 0);
  glb_init(done == 0);
  if (done >= 2) {
    if (n == 0) {
      p = p1 * p2;
    } else {
      p = 2 * p1 * p2;
    }
    // proves that n \in {0,1}
    assert(n >= 0); assert(n <= 1);
  }
}

void main_smaller() { // same as main(), but frontend produces only 8 transitions
  int p; 
  int NONDET;
  glb_init(n >= 0);
  glb_init(done == 0);
  if (done >= 2) {
    if (NONDET) {
      { __ESBMC_atomic_begin();
	    __ESBMC_assume(n == 0);
	    p = p1*p2;
        __ESBMC_atomic_end();
      }
    } else {
      { __ESBMC_atomic_begin();
	    __ESBMC_assume(n != 0);
	    p = 2*p1*p2;
        __ESBMC_atomic_end();
      }
    }
    // proves that n \in {0,1}
    { __ESBMC_atomic_begin();
      assert(n >= 0);
      assert(n <= 1);
      __ESBMC_atomic_end();
    }
  }
}

void thr1() {
  p1 = 1;
  while (n > 1) {
    { __ESBMC_atomic_begin();
      n = n - 1; 
      p1 = 2 * p1;
      __ESBMC_atomic_end();
    }
  }
  done++;
}

void thr2() {
  p2 = 1;
  while (n > 1) {
    { __ESBMC_atomic_begin();
      n = n - 1;
      p2 = 2 * p2;
      __ESBMC_atomic_end();
    }
  }
  done++;
}

// core program that has a non-thread-modular proof:
/*
void main() { 
  glb_init(n >= 0);
  glb_init(done == 0);
  if (done >= 2) {
    assert(n >= 0);
  }
}

void thr1() {
  while (n > 1) {
      n = n - 1; 
  }
  done++;
}

void thr2() {
  while (n > 1) {
      n = n - 1;
  }
  done++;
}
*/
