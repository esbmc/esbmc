extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}
extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }
extern void __VERIFIER_atomic_begin(void);
extern void __VERIFIER_atomic_end(void);

/* Testcase from Threader's distribution. For details see:
   http://www.model.in.tum.de/~popeea/research/threader

   This file is adapted from the example introduced in the paper:
   Thread-Modular Verification for Shared-Memory Programs
   by Cormac Flanagan, Stephen Freund, Shaz Qadeer.
*/

#include <pthread.h>
#undef assert
#define assert(e) if (!(e)) ERROR: reach_error()

int w=0, r=0, x, y;

void __VERIFIER_atomic_take_write_lock() {
  assume_abort_if_not(w==0 && r==0);
  w = 1;
}

void __VERIFIER_atomic_take_read_lock() {
  assume_abort_if_not(w==0);
  r = r+1;
}

void *writer(void *arg) { //writer
  __VERIFIER_atomic_take_write_lock();
  __VERIFIER_atomic_begin();
  x = 3;
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  w = 0;
  __VERIFIER_atomic_end();
  return 0;
}

void *reader(void *arg) { //reader
  int l;
  __VERIFIER_atomic_take_read_lock();
  __VERIFIER_atomic_begin();
  l = x;
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  y = l;
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  int ly = y;
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  int lx = x;
  __VERIFIER_atomic_end();
  assert(ly == lx);
  __VERIFIER_atomic_begin();
  l = r-1;
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  r = l;
  __VERIFIER_atomic_end();
  return 0;
}

int main() {
  pthread_t t1, t2, t3, t4;
  pthread_create(&t1, 0, writer, 0);
  pthread_create(&t2, 0, reader, 0);
  pthread_create(&t3, 0, writer, 0);
  pthread_create(&t4, 0, reader, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  pthread_join(t3, 0);
  pthread_join(t4, 0);
  return 0;
}

