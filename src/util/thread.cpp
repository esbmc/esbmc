#include <util/thread.h>

Thread::Thread()
{
  pthread_mutex_init(&_mutex, nullptr);

  pthread_attr_init(&_attr);
  pthread_attr_setdetachstate(&_attr, PTHREAD_CREATE_DETACHED);
  pthread_attr_setscope(&_attr, PTHREAD_SCOPE_SYSTEM);
}

Thread::~Thread()
{
  pthread_mutex_unlock(&_mutex);
  pthread_mutex_destroy(&_mutex);

  pthread_attr_destroy(&_attr);
}

void *Thread::function(void *ptr)
{
  if(!ptr)
  {
    return nullptr;
  }

  static_cast<Thread *>(ptr)->run();
  pthread_exit(ptr);
  return nullptr;
}

void Thread::start()
{
  pthread_create(&_id, &_attr, Thread::function, this);
  pthread_detach(_id);
}

void Thread::lock()
{
  pthread_mutex_lock(&_mutex);
}

void Thread::unlock()
{
  pthread_mutex_unlock(&_mutex);
}
