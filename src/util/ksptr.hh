
#pragma once

#include <utility>
#include <type_traits>

namespace ksptr
{
typedef std::conditional_t<sizeof(void *) >= sizeof(long), long long, long>
  cnt_t;

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
  struct make
  {
  };

  struct control_block
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
  constexpr sptr() noexcept : cb(nullptr)
  {
  }

  constexpr sptr(const sptr &o) noexcept : cb(o.cb)
  {
    if (cb)
      cb->use_count++;
  }

  constexpr sptr(sptr &&o) noexcept : cb(std::exchange(o.cb, nullptr))
  {
  }

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
