#include <pthread.h> 
int table_master = 1;
int table_slave = 1;
int binlog;
int can_start_m = 0, done_m = 0;
int can_start_s = 0, done_s = 0;
int can_start_c = 0, done_c = 0;
int res1, res2;

void t1_master() {
  __ESBMC_assume(can_start_m >= 1);
  table_master = 3;
  binlog = 1;
  done_m++;
}

void t2_master() {
  __ESBMC_assume(can_start_m >= 1);
  table_master = 5;
  binlog = 2;
  done_m++;
}

void slave() {
  __ESBMC_assume(can_start_s > 0);
  if (binlog <= 1) {
    table_slave = 5;
    table_slave = 3;
  } else {
    table_slave = 3;
    table_slave = 5;
  }
  done_s++;
}

void client_from_master() {
  __ESBMC_assume(can_start_c >= 1);
  // read from the master
  res1 = table_master;
  done_c=done_c+1;
}

void client_from_slave() {
  __ESBMC_assume(can_start_c >= 1);
  // read from the slave
  res2 = table_slave;
  done_c=done_c+1;
}

void main() { // complete code for the main thread
  can_start_m = 2;
  // fork (t1_master || t2_master);
  __ESBMC_assume(done_m >= 2);
  can_start_s = 1;
  // fork slave;
  __ESBMC_assume(done_s >= 1);
  can_start_c = 2;
  // fork client_from_master(res1);
  // fork client_from_slave(res);
  __ESBMC_assume(done_c >= 1);
  assert(res1 == res2);
}
