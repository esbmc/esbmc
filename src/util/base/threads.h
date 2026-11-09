#ifndef UTIL_BASE_THREADS_H
#define UTIL_BASE_THREADS_H

#include <atomic>

/// Whether several threads may touch shared ireps, which only
/// --parallel-solving does. Until it sets this (through threads_scope), ESBMC
/// runs on one thread and the reference counts and locks below skip their
/// atomic operations, as libstdc++ does for __libc_single_threaded. It changes
/// only while one thread runs, and starting and joining the threads orders it
/// with them.
inline bool threads_running = false;

/// c += n, returning the old value.
template <typename T>
inline T counter_add(std::atomic<T> &c, T n, std::memory_order order)
{
  if (threads_running)
    return c.fetch_add(n, order);
  const T old = c.load(std::memory_order_relaxed);
  c.store(old + n, std::memory_order_relaxed);
  return old;
}

/// c -= n, returning the old value.
template <typename T>
inline T counter_sub(std::atomic<T> &c, T n, std::memory_order order)
{
  if (threads_running)
    return c.fetch_sub(n, order);
  const T old = c.load(std::memory_order_relaxed);
  c.store(old - n, std::memory_order_relaxed);
  return old;
}

/// Sets threads_running while it lives: create it before starting threads that
/// touch ireps, symbols or the string pool, and let it go after joining them.
class threads_scope
{
public:
  threads_scope()
  {
    threads_running = true;
  }
  ~threads_scope()
  {
    threads_running = false;
  }
  threads_scope(const threads_scope &) = delete;
  threads_scope &operator=(const threads_scope &) = delete;
};

/// A Mutex that locks only while threads_running. No lock may be held across a
/// change to threads_running.
template <typename Mutex>
class threads_mutex
{
public:
  void lock()
  {
    if (threads_running)
      m.lock();
  }
  void unlock()
  {
    if (threads_running)
      m.unlock();
  }
  void lock_shared()
  {
    if (threads_running)
      m.lock_shared();
  }
  void unlock_shared()
  {
    if (threads_running)
      m.unlock_shared();
  }

private:
  Mutex m;
};

#endif
