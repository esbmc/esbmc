
#pragma once

#include <utility>
#include <type_traits>

namespace ksptr
{
/* A type that is large enough to count any number of copies existing in memory
 * (even on Windows) */
using cnt_t =
  std::conditional_t<sizeof(long) < sizeof(void *), long long, long>;

class control_block /* meta-data together with the concrete data */
{
  template <typename T>
  friend class sptr;

  cnt_t use_count;
};

/**
 * A somewhat dumbed-down shared_ptr implementation, still missing weak pointers
 * and reference/array support. The purpose of this class is to provide
 * shared pointers that
 *
 * (a) do not have an atomic reference count, and
 * (b) do not provide for separate storage of control block and actual data,
 *     which allows to reduce space.
 *
 * This comes with restrictions on T: It must have control_block as a unique
 * base class. Furthermore, since the control_block itself doesn't provide a
 * (virtual) destructor, casting to sptr<control_block> generally isn't a good
 * idea.
 */
template <typename T>
class sptr
{
  static_assert(!std::is_reference_v<T>, "reference types are not supported");
  static_assert(!std::is_array_v<T>, "array types are not supported");
  static_assert(std::is_convertible_v<T *, control_block *>);

  template <typename S>
  friend class sptr;

  template <typename S, typename... Args>
  friend sptr<S> make_shared(Args &&...args);

protected:
  struct make /* tag type to select correct constructor */
  {
  };

  control_block *cb;

  T *ptr() const noexcept
  {
    return static_cast<T *>(cb);
  }

  template <typename... Args>
  explicit sptr(make, Args &&...args) : cb(new T(std::forward<Args>(args)...))
  {
    cb->use_count = 1;
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
    : cb(std::exchange(o.cb, nullptr))
  {
  }

  ~sptr()
  {
    if (cb && !--cb->use_count)
      delete ptr();
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
