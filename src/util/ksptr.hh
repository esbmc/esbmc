
#pragma once

#include <utility>
#include <type_traits>

namespace ksptr
{
/* A type that is large enough to count any number of copies existing in memory
 * (even on Windows) */
typedef std::conditional_t<sizeof(void *) >= sizeof(long), long long, long>
  cnt_t;

/**
 * A somewhat dumbed-down shared_ptr implementation, still missing weak pointers
 * and reference/array support. The purpose of this class is to provide
 * shared pointers that
 * (a) do not have an atomic reference count, and
 * (b) do not provide for separate storage of control block and actual data,
 *     which allows to reduce space.
 */
template <typename T>
class sptr
{
  static_assert(!std::is_reference_v<T>, "reference types are not supported");
  static_assert(!std::is_array_v<T>, "array types are not supported");

  template <typename S>
  friend class sptr;

  template <typename S, typename... Args>
  friend sptr<S> make_shared(Args &&...args);

protected:
  struct make /* tag type to select correct constructor */
  {
  };

  struct control_block /* meta-data together with the concrete data */
  {
    cnt_t use_count;
    void (*dtor)(void *);
    alignas(T) char value[sizeof(T)];

    explicit control_block(void (*dtor)(void *)) : use_count(1), dtor(dtor)
    {
    }

    ~control_block()
    {
      dtor(value);
    }
  };

  control_block *cb;

  T *ptr() const noexcept
  {
    return reinterpret_cast<T *>(cb->value);
  }

  template <typename... Args>
  explicit sptr(make, Args &&...args)
    : cb(new control_block([](void *p) { reinterpret_cast<T *>(p)->~T(); }))
  {
    new (ptr()) T(std::forward<Args>(args)...);
  }

public:
  /* constructor and destructor */

  /* default */
  constexpr sptr() noexcept : cb(nullptr)
  {
  }

  /* copy */
  constexpr sptr(const sptr &o) noexcept : cb(o.cb)
  {
    if (cb)
      cb->use_count++;
  }

  /* move */
  constexpr sptr(sptr &&o) noexcept : cb(std::exchange(o.cb, nullptr))
  {
  }

  /* type-cast from an sptr of type convertible to T */
  template <
    typename S,
    typename = std::enable_if_t<std::is_convertible_v<S *, T *>>>
  constexpr sptr(sptr<S> o) noexcept
    : cb(reinterpret_cast<control_block *>(o.cb))
  {
    o.cb = nullptr;
  }

  ~sptr()
  {
    if (cb && !--cb->use_count)
      delete cb;
  }

  /* swap & assignment operators */

  void swap(sptr &b) noexcept
  {
    std::swap(cb, b.cb);
  }

  friend void swap(sptr &a, sptr &b) noexcept
  {
    a.swap(b);
  }

  sptr &operator=(sptr o) noexcept
  {
    swap(o);
    return *this;
  }

  /* observers */

  cnt_t use_count() const noexcept
  {
    return cb ? cb->use_count : 0;
  }

  T &operator*() const noexcept
  {
    return *get();
  }

  T *operator->() const noexcept
  {
    return get();
  }

  T *get() const noexcept
  {
    return cb ? ptr() : nullptr;
  }

  constexpr operator bool() const noexcept
  {
    return cb != nullptr;
  }

  /* mutators */

  void reset() noexcept
  {
    *this = sptr();
  }
};

template <typename T, typename... Args>
sptr<T> make_shared(Args &&...args)
{
  return sptr<T>(typename sptr<T>::make{}, std::forward<Args>(args)...);
}

} // namespace ksptr
