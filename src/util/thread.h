
#ifndef __THREAD_H__
#define __THREAD_H__

#include <pthread.h>

class Thread
{
public:
  Thread();
  virtual ~Thread();
  virtual void start();
  void lock();
  void unlock();

protected:
  virtual void run() = 0;

private:
  pthread_t _id;
  pthread_mutex_t _mutex;
  pthread_attr_t _attr;

  static void *function(void *ptr);
};

#endif
