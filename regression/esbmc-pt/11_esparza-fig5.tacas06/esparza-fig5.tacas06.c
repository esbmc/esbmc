/* the C program runs correctly (no assertion violation) 
when the two file fields are passed by reference.
*/
// #include <assert.h>

int ns_f1_locked = 0, ns_f2_locked = 0; // 2 boolean flags
int ns_f1_pos, ns_f2_pos;
int lock1 = 0, lock2 = 0;

// INV: (lock1 == 1 ==> ns_f1_locked ==1) /\ (lock2 == 1 ==> ns_f2_locked == 1)
void main() {
  int NONDET;
  /* open(f1_locked, f1_pos); */ ns_f1_locked = 1; ns_f1_pos = 0;
  acquire(lock2);
  while (NONDET) {
    /* open(f2_locked, f2_pos); */ ns_f2_locked = 1; ns_f2_pos = 0;
    release(lock2);
    while (NONDET) {
      acquire(lock2);
      /* rw(f2_locked, f2_pos); */ assert(ns_f2_locked >= 1); ns_f2_pos = ns_f2_pos + 1;
      release(lock2);
      acquire(lock1);
      //l++;l++;l++;
      /* rw(f1_locked, f1_pos); */ assert(ns_f1_locked >= 1); ns_f1_pos = ns_f1_pos + 1;
      release(lock1);
    }
    acquire(lock2);
    /* close(f2_locked, f2_pos); */ assert(ns_f2_locked >= 1); ns_f2_locked = 0;
  }
  acquire(lock1);
  /* close(f1_locked, f1_pos); */ assert(ns_f1_locked >= 1); ns_f1_locked = 0;
}
