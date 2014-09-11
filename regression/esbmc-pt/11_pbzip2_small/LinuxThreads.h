#define pthread_t unsigned long
#define pthread_mutex_t struct { int m_spinlock;  int m_count; }
// #define pthread_cond_t void // Nothing

#define acquire(l) \
  { __blockattribute__((atomic)) \
    assume(l==0); \
    l = 1; \
  }

#define release(l) \
	l = 0

#define pthread_mutex_lock(m_spinlock, m_count) \
  while(1) { \
    acquire(m_spinlock); \
      if (m_count == 0) { \
        m_count = 1; \
        release(m_spinlock); \
        break; \
      } \
  } 

#define pthread_mutex_unlock(m_spinlock, m_count) \
  acquire(m_spinlock); \
  m_count = 0; \
  release(m_spinlock)

#define pthread_cond_wait(m_spinlock, m_count) \
  pthread_mutex_unlock(m_spinlock, m_count); \
  pthread_mutex_lock(m_spinlock, m_count)

#define pthread_cond_signal() // Nothing

#define pthread_cond_broadcast() // Nothing

/* Relevant files from linuxthreads-0.71:
- pthread.h: pthread_t, pthread_mutex_t, pthread_cond_t, PTHREAD_MUTEX_FAST_NP, PTHREAD_MUTEX_INITIALIZER
- spinlock.h: acquire, release
- mutex.c: pthread_mutex_lock, pthread_mutex_unlock
- condvar.c: pthread_cond_wait, pthread_cond_signal, pthread_cond_broadcast
*/ 

/* Relevant files from glibc-2.13/nptl:
?
*/
