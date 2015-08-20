#include <pthread.h>
#include <pthread.h>
int block;
int busy; // boolean flag indicating whether the block has be an allocated to an inode
int inode;
int m_inode=0; // protects the inode
int m_busy=0; // protects the busy flag

thr1(){
  glb_init(inode == busy);
  acquire(m_inode);
  if(inode == 0){
    acquire(m_busy);
    busy = 1;
    release(m_busy);
    inode = 1;
  }
  block = 1;
  assert(block == 1);
  release(m_inode);
}

thr2(){
  acquire(m_busy);
  if(busy == 0){
    block = 0;
    assert(block == 0);
  }
  release(m_busy);
}
 
int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
