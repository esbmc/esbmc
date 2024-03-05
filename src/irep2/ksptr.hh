
#pragma once

#include <utility>
#include <typeinfo>
#include <typeindex>
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

  friend class FreeList;

  union {
    cnt_t use_count;
    control_block *next;
  };
};

class FreeList {
  std::unordered_map<std::type_index, size_t> object_sizes;
  control_block *free_lists[16];
  size_t reused = 0;

  FreeList();
  ~FreeList();

  control_block *& operator[](size_t sz)
  {
    size_t n = (sz + 7) / 8;
    assert(n < 16);
    return free_lists[n];
  }

  control_block ** operator[](const std::type_info &info)
  {
    auto it = object_sizes.find(info);
    return it == object_sizes.end() ? nullptr : &(*this)[it->second];
  }

  static FreeList &singleton() // avoid static init order fiasco
  {
    static FreeList singleton;
    return singleton;
  }

public:
  static control_block * obtain(size_t sz)
  {
    FreeList &single = singleton();
    if (control_block *&f = single[sz])
    {
      single.reused++;
      return std::exchange(f, f->next);
    }
    return nullptr;
  }

  static bool insert(control_block *cb, const std::type_info &info)
  {
    FreeList &single = singleton();
    if (control_block **pf = single[info])
    {
      cb->next = std::exchange(*pf, cb);
      return true;
    }
    return false;
  }

  template <typename T>
  struct register_type0
  {
    register_type0()
    {
      FreeList &single = FreeList::singleton();
      single.object_sizes.try_emplace(typeid(T), sizeof(T));
    }
  };
};

class Arena {

  struct Block {
    Block *next;
    unsigned char data[4096 - sizeof(next) - 16];
  } *head;
  size_t head_alloced;

  Arena() : head(new Block), head_alloced(0)
  {
    head->next = nullptr;
  }

  ~Arena();

  size_t left() const
  {
    return sizeof(Block::data) - head_alloced;
  }

  void *_alloc(size_t n)
  {
    if (left() < n)
    {
      Block *b = new Block;
      b->next = head;
      head = b;
      head_alloced = 0;
      assert(left() >= n);
    }
    void *r = head->data + head_alloced;
    head_alloced = (head_alloced + n + 7) & -8;
    return r;
  }

  static Arena & singleton()
  {
    static Arena singleton;
    return singleton;
  }

public:
  static void *alloc(size_t n);
  static void dealloc(void *) {}
};

extern size_t sptrs_alive, sptrs_max;

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
    make()
    {
      static const FreeList::register_type0<T> registered_type {};
    }
  };

  control_block *cb;

  T *ptr() const noexcept
  {
    return static_cast<T *>(cb);
  }

  template <typename... Args>
  explicit sptr(make, Args &&...args) : cb(FreeList::obtain(sizeof(T)))
  {
    if (!cb)
      cb = static_cast<T *>(Arena::alloc(sizeof(T)));
    new (ptr()) T(std::forward<Args>(args)...);
    assert(typeid(*ptr()) == typeid(T));
    cb->use_count = 1;
    if (++sptrs_alive > sptrs_max)
      sptrs_max = sptrs_alive;
  }

public:
  /* constructor and destructor */

  /* default */
  constexpr sptr() noexcept : cb(nullptr)
  {
    if (++sptrs_alive > sptrs_max)
      sptrs_max = sptrs_alive;
  }

  /* copy */
  constexpr sptr(const sptr &o) noexcept : cb(o.cb)
  {
    if (cb)
      cb->use_count++;
    if (++sptrs_alive > sptrs_max)
      sptrs_max = sptrs_alive;
  }

  /* move */
  constexpr sptr(sptr &&o) noexcept : cb(std::exchange(o.cb, nullptr))
  {
    if (++sptrs_alive > sptrs_max)
      sptrs_max = sptrs_alive;
  }

  /* type-cast from an sptr of type convertible to T */
  template <
    typename S,
    typename = std::enable_if_t<std::is_convertible_v<S *, T *>>>
  constexpr sptr(sptr<S> o) noexcept
    : cb(o ? o.ptr() : nullptr)
  {
    o.cb = nullptr;
    if (++sptrs_alive > sptrs_max)
      sptrs_max = sptrs_alive;
  }

  ~sptr()
  {
    if (cb && !--cb->use_count)
    {
      bool r = FreeList::insert(cb, typeid(*ptr()));
      ptr()->~T();
      if (!r)
        Arena::dealloc(ptr());
    }
    --sptrs_alive;
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
