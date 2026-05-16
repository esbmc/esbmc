#ifndef IREP2_H_
#define IREP2_H_

/** @file irep2.h
 *  Classes and definitions for non-stringy internal representation.
 *
 *  Threading contract: irep2 nodes (type2t, expr2t) are designed for
 *  single-writer / thread-confined construction and rewriting.
 *
 *  Ownership is handled by an intrusive atomic refcount sitting on
 *  every node (irep2t::refcount); irep_container is a raw pointer
 *  that increments on copy, decrements on destruction, and deletes
 *  the pointee when the count drops to zero. The refcount is atomic
 *  so two containers holding the same node can be dropped from
 *  different threads safely, but the *pointee's* state is not — only
 *  one thread at a time may obtain a mutable view (the non-const
 *  irep_container::get() / operator-> / operator*, which detach if
 *  refcount > 1 and otherwise hand back the underlying object).
 *
 *  The CRC cache (irep2t::crc_val) is a single std::atomic<size_t>
 *  with 0 meaning "not yet computed". Readers do an acquire load,
 *  producers compute on a stack local and release-store the result,
 *  so concurrent readers see either 0 (and recompute, getting the
 *  same value) or the final cache entry — never a half-mixed state.
 *
 *  Treat an irep2 tree as owned by a single rewriter at a time;
 *  publish to other threads only once mutation is complete and only
 *  for read-only consumption.
 */

#include <big-int/bigint.hh>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <util/compiler_defs.h>
#include <util/crypto_hash.h>
#include <util/irep_idt.h>
#include <util/irep.h>

// The list of irep2 expression kinds lives in expr_kinds.inc; the
// list of type kinds lives in type_kinds.inc. Every consumer (the
// enum entries, the per-kind forward declarations, the is/to/try_to
// predicate generators, the pretty name tables) #include the matching
// .inc with a redefining IREP2_EXPR / IREP2_TYPE macro.

// Even crazier forward decls,
namespace esbmct
{
template <typename... Args>
class type2t_traits;

/** Type-erased callable reference: non-owning, two pointers wide, no
 *  heap allocation. Replaces the historic `std::function` delegates
 *  used by foreach_operand / foreach_subtype. The contract is a
 *  borrowed reference to the underlying callable — the function_ref
 *  must not outlive whatever it was constructed from.
 *
 *  The unparenthesised primary template is left undefined; only the
 *  R(Args...) specialisation is usable, matching the std::function
 *  surface we are replacing.
 */
template <typename Signature>
class function_ref;

template <typename R, typename... Args>
class function_ref<R(Args...)>
{
public:
  template <
    typename F,
    typename = std::enable_if_t<
      !std::is_same_v<std::decay_t<F>, function_ref> &&
      std::is_invocable_r_v<R, F &, Args...>>>
  function_ref(F &&fn) noexcept
    : invoke_(&function_ref::call<std::remove_reference_t<F>>),
      ctx_(const_cast<void *>(static_cast<const void *>(std::addressof(fn))))
  {
  }

  R operator()(Args... args) const
  {
    if constexpr (std::is_void_v<R>)
      invoke_(ctx_, std::forward<Args>(args)...);
    else
      return invoke_(ctx_, std::forward<Args>(args)...);
  }

private:
  template <typename F>
  static R call(void *p, Args... args)
  {
    if constexpr (std::is_void_v<R>)
      (*static_cast<F *>(p))(std::forward<Args>(args)...);
    else
      return (*static_cast<F *>(p))(std::forward<Args>(args)...);
  }

  R (*invoke_)(void *, Args...);
  void *ctx_;
};

/** Mix a value into a running hash seed.
 *
 *  Traditional golden-ratio magic + shift mix. std::hash is used as
 *  the per-type hasher. There is no std::hash_combine in the standard
 *  library (yet — see WG21 proposals); this 3-line inline helper is
 *  the stdlib-equivalent.
 */
template <class T>
inline void hash_combine(std::size_t &seed, const T &v)
{
  seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace esbmct

class type2t;
class expr2t;
class constant_array2t;
class constant_vector2t;

/** Reference counted container for irep2 nodes.
 *
 *  Holds a raw `T *` to an irep2-derived node and manages its intrusive
 *  refcount: construction increments, destruction decrements, the
 *  pointee is deleted when the count reaches zero. Single allocation
 *  per node — no separate control block, no `shared_ptr` indirection.
 *
 *  The container also implements the historic copy-on-write contract:
 *  the only way to obtain a non-const `T *` / `T &` is via the
 *  non-const `get()` / `operator*` / `operator->`, all of which call
 *  `detach()` first. If the underlying node is shared (refcount > 1),
 *  `detach()` allocates a fresh copy via `T::clone()` and rebinds the
 *  container to that copy; the other holders keep observing the
 *  original. This preserves the value semantics of the legacy
 *  string-irep `irept` while letting most subtree copies be cheap.
 *
 *  Threading. See irep2.h's preamble. The refcount is atomic
 *  (acq/rel on release/destruction, relaxed on construction copy)
 *  but the container is *not* designed for concurrent mutation: under
 *  the single-writer contract there is at most one mutator at a time,
 *  while readers may share the same node.
 */
template <class T>
class irep_container
{
public:
  // Default: empty container, nullptr inside. `noexcept`/`constexpr` to
  // preserve the storage-class properties of the previous design.
  constexpr irep_container() noexcept : ptr_(nullptr)
  {
  }

  // Adopt a freshly-allocated node. The pointee's refcount must be 0
  // (i.e. just-new'd); we bump it to 1. The factory functions in
  // irep2_expr.h / irep2_type.h are the only intended callers of this
  // overload; outside code constructs containers via copy/move.
  explicit irep_container(T *raw) noexcept : ptr_(raw)
  {
    if (ptr_)
      ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
  }

  irep_container(const irep_container &ref) noexcept : ptr_(ref.ptr_)
  {
    if (ptr_)
      ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
  }

  irep_container(irep_container &&ref) noexcept : ptr_(ref.ptr_)
  {
    ref.ptr_ = nullptr;
  }

  ~irep_container()
  {
    release();
  }

  irep_container &operator=(const irep_container &ref) noexcept
  {
    if (this != &ref)
    {
      // Snapshot ref.ptr_ before calling release().  ref may alias a
      // member of an object that release() destroys (e.g.
      // `x = to_array_type(x).subtype` where x holds the only reference
      // to the array): after release() runs, that storage is gone and
      // any further access to ref is use-after-free.  Reading the
      // pointer up-front and then working off the local copy keeps us
      // correct regardless of whether the compiler caches or reloads
      // ref.ptr_ (gcc on Linux happens to cache it, AppleClang on macOS
      // reloads and crashes).
      T *new_ptr = ref.ptr_;
      if (new_ptr)
        new_ptr->refcount.fetch_add(1, std::memory_order_relaxed);
      release();
      ptr_ = new_ptr;
    }
    return *this;
  }

  irep_container &operator=(irep_container &&ref) noexcept
  {
    if (this != &ref)
    {
      // Same use-after-free concern as the copy-assign path: snapshot
      // and null out ref before release(), so even if release() ends
      // up destroying the storage backing ref we never re-read it.
      T *new_ptr = ref.ptr_;
      ref.ptr_ = nullptr;
      release();
      ptr_ = new_ptr;
    }
    return *this;
  }

  // Const accessors: hand out the pointee without detaching, since
  // they're const operations. Prefer these whenever you only need to
  // read the node — they are O(1) atomic-free pointer loads.
  [[nodiscard]] const T &operator*() const noexcept
  {
    return *ptr_;
  }

  [[nodiscard]] const T *operator->() const noexcept
  {
    return ptr_;
  }

  [[nodiscard]] const T *get() const noexcept
  {
    return ptr_;
  }

  // Non-const accessors detach: if the pointee is shared (refcount > 1)
  // we clone into a fresh refcount-1 object and rebind, then invalidate
  // the CRC cache on the (now uniquely-owned) copy. **WARNING:** even
  // when no actual mutation follows, calling these allocates+copies on
  // every shared node touch — favour the const overloads above when a
  // read is enough.
  T *get()
  {
    detach();
    // Single-writer contract: relaxed is enough to invalidate the
    // cache; the next crc() recomputation will publish via release.
    ptr_->crc_val.store(0, std::memory_order_relaxed);
#ifndef NDEBUG
    // Stamp the current thread as the writer of this (now uniquely
    // owned) node; if another thread is already inside a mutable
    // access on the same node, mark_writer asserts.
    ptr_->mark_writer();
#endif
    return ptr_;
  }

  T &operator*()
  {
    return *get();
  }

  T *operator->()
  {
    return get();
  }

  explicit operator bool() const noexcept
  {
    return ptr_ != nullptr;
  }

  void reset() noexcept
  {
    release();
    ptr_ = nullptr;
  }

  void detach()
  {
    if (!ptr_)
      return;
    // Acquire so we observe whatever state the owner of the prior
    // (about-to-be-replaced) reference flushed before storing here.
    // Almost always 1 — only shared subtrees go down the clone path.
    if (ptr_->refcount.load(std::memory_order_acquire) == 1)
      return;
    // Shared: clone into a fresh refcount-0 node and adopt it.
    *this = ptr_->clone();
  }

  friend void swap(irep_container &a, irep_container &b) noexcept
  {
    T *tmp = a.ptr_;
    a.ptr_ = b.ptr_;
    b.ptr_ = tmp;
  }

  void swap(irep_container &b) noexcept
  {
    using std::swap;
    swap(*this, b);
  }

  irep_container simplify() const
  {
    const T *foo = get();
    return foo->simplify();
  }

  size_t crc() const
  {
    const T *foo = get();
    // Acquire ordering on the load so a non-zero cached value also
    // synchronises with the writer that produced it (do_crc()'s release
    // store), and we observe whatever node state went into computing it.
    if (size_t cached = foo->crc_val.load(std::memory_order_acquire);
        cached != 0)
      return cached;
    return foo->do_crc();
  }

  /* Provide comparison operators here as inline friends so they don't pollute
   * the outer namespace; this reduces clutter when there are error messages
   * about these infix operators. It also means that no user-defined
   * conversions are considered unless at least one operand has the type of
   * this class or is derived from it. This is usually wanted since supplying
   * those conversions means someone else has to care about comparing whatever
   * values they potentially convert...
   *
   * This implementation assumes that the type T is totally ordered.
   *
   * TODO: when switching to >= C++20, replace these with only operator== and
   * operator<=>
   */

  friend bool operator==(const irep_container &a, const irep_container &b)
  {
    if (same(a, b))
      return true;

    if (!a || !b)
      return false;

    return *a == *b; // different pointees could still compare equal
  }

  friend bool operator!=(const irep_container &a, const irep_container &b)
  {
    return !(a == b);
  }

  friend bool operator<(const irep_container &a, const irep_container &b)
  {
    if (!b)
      return false; // If b is nil, nothing can be lower
    if (!a)
      return true; // nil is lower than non-nil

    if (same(a, b))
      return false;

    return *a < *b;
  }

  friend bool operator<=(const irep_container &a, const irep_container &b)
  {
    return !(a > b);
  }

  friend bool operator>=(const irep_container &a, const irep_container &b)
  {
    return !(a < b);
  }

  friend bool operator>(const irep_container &a, const irep_container &b)
  {
    return b < a;
  }

private:
  static bool same(const irep_container &a, const irep_container &b) noexcept
  {
    // Direct pointer equality is fine here because both pointers were
    // obtained from `new` (or are nullptr), so they are valid objects
    // of the same type and the comparison is well-defined.
    return a.ptr_ == b.ptr_;
  }

  // Drop our reference. Release ordering pairs with the acquire in
  // detach()/destruction on whichever container observes refcount==1
  // and is about to mutate or delete the pointee; this makes any
  // prior writes through *this happen-before that observer's access.
  void release() noexcept
  {
    if (!ptr_)
      return;
    unsigned int prev = ptr_->refcount.fetch_sub(1, std::memory_order_release);
    if (prev == 1)
    {
      // Acquire fence prevents the delete (and any side-effects of
      // the destructor) from being reordered before the final
      // fetch_sub, ensuring we see the prior owner's writes.
      std::atomic_thread_fence(std::memory_order_acquire);
      delete ptr_;
    }
#ifndef NDEBUG
    else if (prev == 2)
    {
      // We were one of exactly two owners and just dropped out; the
      // remaining sole owner is free to mutate. Clear the writer
      // stamp so they get a clean slate (the next non-const get()
      // will re-stamp).
      ptr_->clear_writer();
    }
#endif
  }

  T *ptr_;
};

typedef irep_container<type2t> type2tc;
typedef irep_container<expr2t> expr2tc;

/** Allocate a fresh irep2 node and adopt it into an `irep_container`.
 *
 *  The single canonical construction path for the intrusive refcount
 *  scheme. `T` must derive from an irep2 base (`type2t` / `expr2t`)
 *  that exposes `base_type` as its method-side base; the returned
 *  container is `irep_container<T::base_type>` (i.e. `type2tc` or
 *  `expr2tc`).
 *
 *  Single allocation, no separate control block, exception-safe (if
 *  any argument constructor throws, the `new` hasn't run yet; if the
 *  `new` itself throws, no leak; the only ordering between `new`
 *  succeeding and the container adopting it is straight-line code
 *  inside this function).
 *
 *  Callers should prefer this helper over a raw `new`-into-container
 *  so the allocation site stays in one place.
 */
template <class T, class... Args>
inline irep_container<typename T::base_type> make_irep(Args &&...args)
{
  return irep_container<typename T::base_type>(
    new T(std::forward<Args>(args)...));
}

typedef std::pair<std::string, std::string> member_entryt;
typedef std::list<member_entryt> list_of_memberst;

/** Base class for every irep2 node. Carries the intrusive refcount
 *  used by irep_container; the cell sits inside the node itself.
 *  Container construction increments; container destruction
 *  decrements and deletes when the count reaches zero. See irep2.h's
 *  threading-contract preamble for the ordering rules.
 */
class irep2t
{
public:
  irep2t() noexcept
    : refcount(0)
#ifndef NDEBUG
      ,
      writer_thread(0)
#endif
  {
  }
  // Copy constructor must NOT propagate the refcount: a fresh copy is
  // a brand new object that no container has yet adopted. The single
  // copy site that matters is clone(), which always wraps the result
  // in an irep_container immediately afterwards.
  irep2t(const irep2t &) noexcept
    : refcount(0)
#ifndef NDEBUG
      ,
      writer_thread(0)
#endif
  {
  }
  // Refcount is per-object identity; assignment is meaningless on the
  // base. Subclasses' copy-assignment operators do not call into here.
  irep2t &operator=(const irep2t &) = delete;
  virtual ~irep2t() = default;

  mutable std::atomic<unsigned int> refcount;

#ifndef NDEBUG
  // Debug-only writer-thread stamp. The threading contract says irep2
  // nodes are single-writer; on the first mutable access through an
  // irep_container, the container stamps this slot with a hash of the
  // current thread id (0 means "no writer"). Subsequent mutable
  // accesses compare — a mismatch means two threads are trying to
  // mutate the same uniquely-owned node, which is the contract
  // violation we want to catch. The stamp clears when the refcount
  // drops back to 1 in release(), so a single-owner-handoff across
  // threads (publish, then a new owner mutates) works correctly.
  // Compiled out entirely under NDEBUG. uintptr_t is used (rather
  // than std::thread::id directly) so the atomic is guaranteed
  // lock-free on every platform we target.
  mutable std::atomic<std::uintptr_t> writer_thread;

  static std::uintptr_t current_thread_tag() noexcept
  {
    auto h = std::hash<std::thread::id>{}(std::this_thread::get_id());
    // 0 is the "no writer" sentinel; if the hash collides with 0
    // (vanishingly unlikely but possible), bump it.
    return h == 0 ? 1 : static_cast<std::uintptr_t>(h);
  }

  void mark_writer() const noexcept
  {
    std::uintptr_t me = current_thread_tag();
    std::uintptr_t prev = writer_thread.load(std::memory_order_acquire);
    if (prev == me)
      return; // already mine, common fast path
    if (prev == 0)
    {
      // Race-free attempt to claim the slot. If we lose the CAS to a
      // concurrent writer we fall through to the mismatch assert
      // below, which is the contract violation.
      if (writer_thread.compare_exchange_strong(
            prev, me, std::memory_order_release, std::memory_order_acquire))
        return;
    }
    assert(
      prev == me &&
      "irep2 single-writer contract violated: a second thread tried to "
      "mutate a node already being mutated by another thread");
  }

  void clear_writer() const noexcept
  {
    writer_thread.store(0, std::memory_order_release);
  }
#endif
};

/** Base class for all types.
 *  Contains only a type identifier enumeration - for some types (such as bool,
 *  or empty,) there's no need for any significant amount of data to be stored.
 */
class type2t : public irep2t
{
public:
  /** Enumeration identifying each sort of type. Driven by
   *  type_kinds.inc; see that file's header for the IREP2_TYPE(kind,
   *  pretty_name) contract. */
  enum type_ids
  {
#define IREP2_TYPE(kind, pretty) kind##_id,
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
    end_type_id
  };

  /* Define default traits */
  typedef typename esbmct::type2t_traits<> traits;

  /** Symbolic type exception class.
   *  To be thrown when attempting to fetch the width of a symbolic type, such
   *  as empty or code. Caller will have to worry about what to do about that.
   */
  class symbolic_type_excp
  {
  public:
    const char *what() const noexcept
    {
      return "symbolic type encountered";
    }
  };

  // Non-owning callable references. The historic typedefs were
  // std::function<void(...)>, which allocated for any non-trivially-
  // copyable lambda capture and indirected through the std::function
  // type-erasure. function_ref is two pointers wide and inlines into
  // the caller's stack frame.
  typedef esbmct::function_ref<void(const type2tc &)> const_subtype_delegate;
  typedef esbmct::function_ref<void(type2tc &)> subtype_delegate;

protected:
  /** Primary constructor.
   *  @param id Type ID of type being constructed
   */
  type2t(type_ids id);

  /** Copy constructor */
  type2t(const type2t &ref);

  virtual void foreach_subtype_impl_const(const_subtype_delegate &t) const = 0;
  virtual void foreach_subtype_impl(subtype_delegate &t) = 0;

  // Non-virtual switch-based dispatchers (issue #4560 scaffolding).
  void foreach_subtype_impl_const_v2(const_subtype_delegate &t) const;
  void foreach_subtype_impl_v2(subtype_delegate &t);

public:
  // Provide base / container types for some templates stuck on top:
  typedef type2tc container_type;
  typedef type2t base_type;

  virtual ~type2t() = default;

  /** Fetch bit width of this type.
   *  For a particular type, calculate its size in a bit representation of
   *  itself. May throw various exceptions depending on whether this operation
   *  is viable - for example, for symbol types, infinite sized or dynamically
   *  sized arrays.
   *
   *  Note that the bit width is _not_ the same as the ansi-c byte model
   *  representation of this type.
   *
   *  @throws symbolic_type_excp
   *  @throws array_type2t::inf_sized_array_excp
   *  @throws array_type2t::dyn_sized_array_excp
   *  @return Size of types byte representation, in bits
   */
  virtual unsigned int get_width() const = 0;

  bool operator==(const type2t &ref) const;
  bool operator!=(const type2t &ref) const;
  bool operator<(const type2t &ref) const;

  /** Produce a string representation of type.
   *  Takes body of the current type and produces a human readable
   *  representation. Similar to the string-irept's pretty method, although a
   *  different format.
   *  @param indent Number of spaces to indent lines by in the output
   *  @return String obj containing representation of this object
   */
  std::string pretty(unsigned int indent = 0) const;

  /** Dump object string representation to stdout.
   *  This take the output of the pretty method, and dumps it to stdout. To be
   *  used for debugging and when single stepping in gdb.
   *  @see pretty
   */
  DUMP_METHOD void dump() const;

  /** Produce a checksum/hash of the current object.
   *  Takes current object and produces a lossy digest of it. Originally used
   *  crc32, now uses a more hacky but faster hash function. For use in hash
   *  objects.
   *  @see do_crc
   *  @return Digest of the current type.
   */
  size_t crc() const;

  /** Perform checked invocation of cmp method.
   *  Takes reference to another type - if they have the same type id, invoke
   *  the cmp function and return its result. Otherwise, return false. Using
   *  this method ensures thatthe implementer of cmp knows the reference it
   *  operates on is on the same type as itself.
   *  @param ref Reference to type to compare this object against
   *  @return True if types are the same, false otherwise.
   */
  bool cmpchecked(const type2t &ref) const;

  /** Perform checked invocation of lt method.
   *  Identical to cmpchecked, except with the lt method.
   *  @see cmpchecked
   *  @param ref Reference to type to measure this against.
   *  @return 0 if types are the same, 1 if this > ref, -1 if ref > this.
   */
  int ltchecked(const type2t &ref) const;

  /** Virtual method to compare two types.
   *  To be overridden by an extending type; assumes that itself and the
   *  parameter are of the same extended type. Call via cmpchecked.
   *  @see cmpchecked
   *  @param ref Reference to (same class of) type to compare against
   *  @return True if types match, false otherwise
   */
  virtual bool cmp(const type2t &ref) const = 0;

  /** Virtual method to compare two types.
   *  To be overridden by an extending type; assumes that itself and the
   *  parameter are of the same extended type. Call via cmpchecked.
   *  @see cmpchecked
   *  @param ref Reference to (same class of) type to compare against
   *  @return 0 if types are the same, 1 if this > ref, -1 if ref > this.
   */
  virtual int lt(const type2t &ref) const;

  /** Extract a list of members from type as strings.
   *  Produces a list of pairs, mapping a member name to a string value. Used
   *  in the body of the pretty method.
   *  @see pretty
   *  @param indent Number of spaces to indent output strings with, if multiline
   *  @return list of name:value pairs.
   */
  virtual list_of_memberst tostring(unsigned int indent) const = 0;

  /** Perform crc operation accumulating into parameter.
   *  Performs the operation of the crc method, but overridden to be specific to
   *  a particular type. Accumulates data into the hash object parameter.
   *  @see cmp
   *  @param seed Hash to accumulate hash data into.
   *  @return Hash value
   */
  virtual size_t do_crc() const;

  /** Perform hash operation accumulating into parameter.
   *  Feeds data as appropriate to the type of the expression into the
   *  parameter, to be hashed. Like crc and do_crc, but for some other kind
   *  of hash scenario.
   *  @see cmp
   *  @see crc
   *  @see do_crc
   *  @param hash Object to accumulate hash data into.
   */
  virtual void hash(crypto_hash &hash) const;

  /** Clone method. Self explanatory.
   *  @return New container, containing a duplicate of this object.
   */
  virtual type2tc clone() const = 0;

  // Please see the equivalent methods in expr2t for documentation
  template <typename T>
  void foreach_subtype(T &&t) const
  {
    const_subtype_delegate wrapped(t);
    foreach_subtype_impl_const(wrapped);
  }

  template <typename T>
  void Foreach_subtype(T &&t)
  {
    subtype_delegate wrapped(t);
    foreach_subtype_impl(wrapped);
  }

  // Non-virtual switch-based dispatchers (issue #4560 scaffolding).
  // These coexist with the virtual methods and will replace them once all
  // kinds are migrated to the flat struct layout.
  bool cmp_v2(const type2t &ref) const;
  int lt_v2(const type2t &ref) const;
  type2tc clone_v2() const;
  size_t do_crc_v2() const;
  void hash_v2(crypto_hash &hash) const;
  list_of_memberst tostring_v2(unsigned int indent) const;
  unsigned int get_width_v2() const;

  /** Instance of type_ids recording this types type. */
  // XXX XXX XXX this should be const
  type_ids type_id;

  // CRC cache: 0 means "not yet computed". Atomic so concurrent readers
  // see either the prior value or the fresh one — never a torn read.
  // Writers respect the single-writer contract documented at the top of
  // this header; the cache is therefore safe to set without a lock as
  // long as readers and the (single) writer share happens-before via the
  // atomic.
  mutable std::atomic<size_t> crc_val;
};

/** Fetch identifying name for a type.
 *  I.E., this is the class of the type, what you'd get if you called type.id()
 *  with the old stringy irep. Ideally this should be a class method, but as it
 *  was added as a hack I haven't got round to it yet.
 *  @param type Type to fetch identifier for
 *  @return String containing name of type class.
 */
std::string get_type_id(const type2t &type);

/** Fetch identifying name for a type.
 *  Just passes through to type2t accepting function with the same name.
 *  @param type Type to fetch identifier for
 *  @return String containing name of type class.
 */
static inline std::string get_type_id(const type2tc &type)
{
  return get_type_id(*type);
}

/** Base class for all expressions.
 *  In this base, contains an expression id used for distinguishing different
 *  classes of expr, in addition we have a type as all exprs should have types.
 */
class expr2t : public irep2t
{
public:
  /** Enumeration identifying each sort of expr.
   */
  enum expr_ids
  {
// The single source of truth for the kind list lives in
// expr_kinds.inc; see that file's header for the IREP2_EXPR(kind,
// pretty_name) contract.
#define IREP2_EXPR(kind, pretty) kind##_id,
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
    end_expr_id
  };

  /** Type for list of constant expr operands */
  typedef std::list<const expr2tc *> expr_operands;
  /** Type for list of non-constant expr operands */
  typedef std::list<expr2tc *> Expr_operands;

  // Non-owning callable references; see commentary on type2t's variants.
  typedef esbmct::function_ref<void(const expr2tc &)> const_op_delegate;
  typedef esbmct::function_ref<void(expr2tc &)> op_delegate;

protected:
  /** Primary constructor.
   *  @param type Type of this new expr
   *  @param id Class identifier for this new expr
   */
  expr2t(const type2tc &type, expr_ids id);
  /** Copy constructor */
  expr2t(const expr2t &ref);

  virtual void foreach_operand_impl_const(const_op_delegate &expr) const;
  virtual void foreach_operand_impl(op_delegate &expr);

  // Non-virtual switch-based dispatchers (issue #4560 scaffolding).
  void foreach_operand_impl_const_v2(const_op_delegate &expr) const;
  void foreach_operand_impl_v2(op_delegate &expr);

public:
  // Provide base / container types for some templates stuck on top:
  typedef expr2tc container_type;
  typedef expr2t base_type;

  virtual ~expr2t() = default;

  /** Clone method. Self explanatory. */
  virtual expr2tc clone() const;

  /* These are all self explanatory */
  bool operator==(const expr2t &ref) const;
  bool operator<(const expr2t &ref) const;
  bool operator!=(const expr2t &ref) const;

  /** Perform type-checked call to lt method.
   *  Checks that this object and the one we're comparing against have the same
   *  expr class, so that the lt method can assume it's working on objects of
   *  the same type.
   *  @see type2t::ltchecked
   *  @param ref Expression object we're comparing this object against.
   *  @return 0 If exprs are the same, 1 if this > ref, -1 if ref > this.
   */
  int ltchecked(const expr2t &ref) const;

  /** Produce textual representation of this expr.
   *  Like the stringy-irep's pretty method, this takes the current object and
   *  produces a textual representation that can be read by a human to
   *  understand what's going on.
   *  @param indent Number of spaces to indent the output string lines by
   *  @return String object containing textual expr representation.
   */
  std::string pretty(unsigned int indent = 0) const;

  /** Write textual representation of this object to stdout.
   *  For use in debugging - dumps the output of the pretty method to stdout.
   *  Can either be used in portion of code, or more commonly called from gdb.
   */
  DUMP_METHOD void dump() const;

  /** Calculate a hash/digest of the current expr.
   *  For use in hash data structures; used to be a crc32, but is now a 16 bit
   *  hash function generated by myself to be fast. May not have nice
   *  distribution properties, but is at least fast.
   *  @return Hash value of this expr
   */
  size_t crc() const;

  /** Perform comparison operation between this and another expr.
   *  Overridden by subclasses of expr2t to compare different members of this
   *  and the passed in object. Assumes that the passed in object is the same
   *  class type as this; Should be called via operator==, which will do that
   *  check automagically.
   *  @see type2t::cmp
   *  @param ref Expr object to compare this against
   *  @return True if objects are the same; false otherwise.
   */
  virtual bool cmp(const expr2t &ref) const;

  /** Compare two expr objects.
   *  Overridden by subclasses - takes two expr objects (this and ref) of the
   *  same type, and compares them, in the same manner as memcmp. The assumption
   *  that the objects are of the same type means lt should be called via
   *  ltchecked to check for different expr types.
   *  @see type2t::lt
   *  @param ref Expr object to compare this against
   *  @return 0 If exprs are the same, 1 if this > ref, -1 if ref > this.
   */
  virtual int lt(const expr2t &ref) const;

  /** Convert fields of subclasses to a string representation.
   *  Used internally by the pretty method - creates a list of pairs
   *  representing the fields in the subclass. Each pair is a pair of strings
   *  of the form fieldname : value. The value may be multiline, in which case
   *  the new line will have at least indent number of indenting spaces.
   *  @param indent Number of spaces to indent multiline output by
   *  @return list of string pairs, of form fieldname:value
   */
  virtual list_of_memberst tostring(unsigned int indent) const;

  /** Perform digest/hash function on expr object.
   *  Takes all fields in this exprs and adds them to the passed in hash object
   *  to compute an expression-hash. Overridden by subclasses.
   *  @param seed Hash to accumulate expression data into.
   *  @return Hash value
   */
  virtual size_t do_crc() const;

  /** Perform hash operation accumulating into parameter.
   *  Feeds data as appropriate to the type of the expression into the
   *  parameter, to be hashed. Like crc and do_crc, but for some other kind
   *  of hash scenario.
   *  @see cmp
   *  @see crc
   *  @see do_crc
   *  @param hash Object to accumulate hash data into.
   */
  virtual void hash(crypto_hash &hash) const;

  /** Fetch a sub-operand.
   *  These can come out of any field that is an expr2tc, or contains them.
   *  No particular numbering order is promised.
   */
  virtual const expr2tc *get_sub_expr(size_t idx) const;

  /** Fetch a sub-operand. Non-const version.
   *  These can come out of any field that is an expr2tc, or contains them.
   *  No particular numbering order is promised.
   */
  virtual expr2tc *get_sub_expr_nc(size_t idx);

  /** Count the number of sub-exprs there are.
   */
  virtual size_t get_num_sub_exprs() const;

  /** Simplify an expression.
   *  Similar to simplification in the string-based irep, this generates an
   *  expression with any calculations or operations that can be simplified,
   *  simplified. In contrast to the old form though, this creates a new expr
   *  if something gets simplified, just to make it clear exactly what's
   *  going on.
   *  @return Either a nil expr (null pointer contents) if nothing could be
   *          simplified or a simplified expression.
   */
  expr2tc simplify() const;

  /** Simplify with reassociation suppression.
   *  When @p suppress_reassoc is true, the chain-root reassociation step
   *  is skipped throughout the subtree (the flag propagates to every
   *  descendant, including those reached through non-chain ops like
   *  modulus). Used by simplify_no_reassoc to walk a subtree with
   *  peepholes only, e.g. on a freshly-rebuilt chain.
   *  External callers should prefer simplify_no_reassoc in
   *  @c util/expr_reassociate.h over calling this overload directly. */
  expr2tc simplify(bool suppress_reassoc) const;

  /** expr-specific simplification methods.
   *  By default, an expression can't be simplified, and this method returns
   *  a nil expression to show that. However if simplification is possible, the
   *  subclass overrides this and if it can simplify its operands, returns a
   *  new simplified expression. It should attempt to modify itself (it's
   *  const).
   *
   *  If simplification failed the first time around, the simplify method will
   *  simplify this expressions individual operands,
   *  and will then call an expr with the simplified operands to see if it's now
   *  become simplifiable. This call occurs whether or not any operands were
   *  actually simplified, see below.
   *
   *  The 'second' parameter can be used to avoid invoking expensive attempts
   *  to simplify an expression more than once - on the first call to
   *  do_simplify this parameter will be false, then on the second it's be true,
   *  allowing method implementation to save the expensive stuff until all of
   *  its operands have certainly been simplified.
   *
   *  Currently simplification does some things that it shouldn't: pointer
   *  arithmetic for example. I'm not sure where this can be relocated to
   *  though.
   *  @return expr2tc A nil expression if no simplifcation could occur, or a new
   *          simplified object if it can.
   */
  [[nodiscard]] virtual expr2tc do_simplify() const;

  /** Indirect, abstract operand iteration.
   *
   *  Provide a lambda-based accessor equivalent to the forall_operands2 macro
   *  where anonymous code (actually a delegate?) gets run over each operand
   *  expression. Because the full type of the expression isn't known by the
   *  caller, and each delegate is it's own type, we need to wrap it in a
   *  std::function before funneling it through a virtual function.
   *
   *  For the purpose of this method, an operand is another instance of an
   *  expr2tc. This means the delegate will be called on any expr2tc field of
   *  the expression, in the order they appear in the traits. For a vector of
   *  expressions, the delegate will be called for each element, in order.
   *
   *  The uncapitalized version is const; the capitalized version is non-const
   *  (and so one needs to .get() a mutable expr2t pointer when calling). When
   *  modifying operands, preserving type correctness is imperative.
   *
   *  @param t A delegate to be called for each expression operand; must have
   *           a type of void f(const expr2tc &)
   */
  template <typename T>
  void foreach_operand(T &&t) const
  {
    const_op_delegate wrapped(t);
    foreach_operand_impl_const_v2(wrapped);
  }

  template <typename T>
  void Foreach_operand(T &&t)
  {
    op_delegate wrapped(t);
    foreach_operand_impl_v2(wrapped);
  }

  // Non-virtual switch-based dispatchers (issue #4560 scaffolding).
  // These coexist with the virtual methods and will replace them once all
  // kinds are migrated to the flat struct layout.
  bool cmp_v2(const expr2t &ref) const;
  int lt_v2(const expr2t &ref) const;
  expr2tc clone_v2() const;
  size_t do_crc_v2() const;
  void hash_v2(crypto_hash &hash) const;
  list_of_memberst tostring_v2(unsigned int indent) const;
  const expr2tc *get_sub_expr_v2(size_t idx) const;
  expr2tc *get_sub_expr_nc_v2(size_t idx);
  size_t get_num_sub_exprs_v2() const;
  [[nodiscard]] expr2tc do_simplify_v2() const;

  /** Instance of expr_ids recording tihs exprs type. */
  const expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  type2tc type;

  // CRC cache; see commentary on type2t::crc_val.
  mutable std::atomic<size_t> crc_val;
};

inline bool is_nil_expr(const expr2tc &exp)
{
  return exp.get() == nullptr;
}

inline bool is_nil_type(const type2tc &t)
{
  return t.get() == nullptr;
}

// For boost multi-index hashing,
inline std::size_t hash_value(const expr2tc &expr)
{
  return expr.crc();
}

/** Fetch string identifier for an expression.
 *  Returns the class name of the expr passed in - this is equivalent to the
 *  result of expr.id() in old stringy irep. Should ideally be a method of
 *  expr2t, but haven't got around to moving it yet.
 *  @param expr Expression to operate upon
 *  @return String containing class name of expression.
 */
std::string get_expr_id(const expr2t &expr);

/** Fetch string identifier for an expression.
 *  Like the expr2t equivalent with the same name, but de-ensapculates an
 *  expr2tc.
 */
static inline std::string get_expr_id(const expr2tc &expr)
{
  return get_expr_id(*expr);
}

/** Template for providing templated methods to irep classes (type2t/expr2t).
 *
 *  What this does: we give irep_methods2 a type trait record that contains
 *  a std::tuple, the elements of which describe each field in the class
 *  we're operating on. For each field we get:
 *
 *    - The type of the field
 *    - The class that field is part of
 *    - A pointer offset to that field.
 *
 *  What this means, is that we can @a type @a generically access a member
 *  of a class from within the template, without knowing what type it is,
 *  what its name is, or even what type contains it.
 *
 *  We can then use that to make all the boring methods of ireps type
 *  generic too. For example: we can make the comparision method by accessing
 *  each field in the class we're dealing with, passing them to another
 *  function to do the comparison (with the type resolved by templates or
 *  via overloading), and then inspecting the output of that.
 *
 *  In fact, we can make type generic implementations of all the following
 *  methods in expr2t: clone, tostring, cmp, lt, do_crc, hash.
 *  Similar methods, minus the operands, can be made generic in type2t.
 *
 *  So, that's what these templates provide; an irep class can be made by
 *  inheriting from this template, telling it what class it'll end up with,
 *  and what to subclass from, and what the fields in the class being derived
 *  from look like. This means we can construct a type hierarchy with whatever
 *  inheritance we like and whatever fields we like, then latch irep_methods2
 *  on top of that to implement all the anoying boring boilerplate code.
 *
 *  ----
 *
 *  In addition, we also define container types for each irep, which is
 *  essentially a type-safeish wrapper around a std::shared_ptr (i.e.,
 *  reference counter). One can create a new irep with syntax such as:
 *
 *    foo2tc bar(type, operand1, operand2);
 *
 *  One can transparently access the irep fields through dereference, such as:
 *
 *    bar->operand1 = 0;
 *
 *  This all replicates the CBMC expression situation, but with the addition
 *  of types.
 *
 *  ----
 *
 *  The following functions can be used to inspect an irep2 object:
 *
 *    is_${suffix}()
 *    to_${suffix}()
 *
 *  For expr2tc the suffix is the name of the class, while for type2t it is the
 *  name of the class without the trailing "2t", e.g.
 *
 *    is_bool_type(type)
 *    to_constant_int2t(expr)
 *
 *  The to_* functions return a (const) reference for a (const) expr2tc or
 *  type2tc parameter. The non-const versions perform a so-called "detach"
 *  operation, which ensures that the to-be-modified object is not referenced by
 *  any other irep2 terms in use. This detach operation is explained in more
 *  detail in the comment about irep_container. Because const-ness is used to
 *  decide whether to detach or not, when working with irep2 it is *critical*
 *  that const_cast<>() is used only where it's safe to. Best practice is to
 *  put a formal safety proof into the comment about const_cast usage.
 *
 *  The above functions are defined by type_macros and expr_macros in the
 *  respective irep2 header.
 *
 *  ----
 *
 *  The traits defined here are used to generically implement the functions
 *  operating on a type2t's or an expr2t's fields, like .dump() and the
 *  iterators foreach_subtype() and foreach_operand().
 *
 *  (The required traits hacks need cleaning up too).
 */
namespace esbmct
{
/** Maximum number of fields to support in expr2t subclasses. This value
 *  controls the types of any arrays that need to consider the number of
 *  fields.
 *  I've yet to find a way of making this play nice with the new variardic
 *  way of defining ireps. */
const unsigned int num_type_fields = 6;

/** Record for properties of an irep field.
 *  This type records, for any particular field:
 *    * It's type
 *    * The class that it's a member of
 *    * A class pointer to this field
 *  The aim being that we have enough information about the field to
 *  manipulate it without any further traits. */
template <typename R, typename C, R C::*v>
class field_traits
{
public:
  typedef R result_type;
  typedef C source_class;
  typedef R C::*membr_ptr;
  static constexpr membr_ptr value = v;
};

template <typename R, typename C, R C::*v>
constexpr
  typename field_traits<R, C, v>::membr_ptr field_traits<R, C, v>::value;

/** Trait class for type2t ireps.
 *  This takes a list of field traits and puts it in a tuple, with the record
 *  for the type_id field (common to all type2t's) put at the front. The
 *  `fields` tuple drives the per-field fold expressions in irep_methods2;
 *  it is the single source of truth for which fields participate in
 *  cmp/lt/crc/hash/operand-iteration. */
template <typename... Args>
class type2t_traits
{
public:
  typedef field_traits<type2t::type_ids, type2t, &type2t::type_id>
    type_id_field;
  typedef std::tuple<type_id_field, Args...> fields;
  typedef type2t base2t;
};

// Declaration of irep and type methods templates. The class walks
// `traits::fields` (a std::tuple of field_traits entries) via fold
// expressions in a single non-recursive class.
template <class derived, class baseclass, typename traits>
class irep_methods2;
template <class derived, class baseclass, typename traits>
class type_methods2;

/** Definition of irep methods template.
 *
 *  @param derived The inheritor class, like add2t
 *  @param baseclass Class containing fields for methods to be defined over
 *  @param traits Type traits for baseclass
 *
 *  A typical irep inheritance looks like this:
 *
 *    b   Base class, such as type2t or expr2t
 *    d   Data class, containing storage fields for ireps
 *    M   irep_methods2 — implements the boilerplate methods (cmp/lt/
 *        clone/crc/hash/tostring/operand iteration) by walking
 *        traits::fields with a fold expression
 *    t   Top level class such as add2t
 *
 *  Each method body is a single fold over the trait list — no inheritance
 *  recursion, no per-field chain levels, no Boost.MP11. Adding a new node
 *  needs the data class to declare its fields and a fields tuple in its
 *  traits; the methods are inherited unchanged.
 */
template <class derived, class baseclass, typename traits>
class irep_methods2 : public baseclass
{
public:
  typedef typename baseclass::base_type base2t;
  typedef irep_container<base2t> base_container2tc;

  template <typename... Args>
  irep_methods2(const Args &...args) : baseclass(args...)
  {
  }

  // Copy constructor. Construct from derived ref rather than just
  // irep_methods2, because the template above will be able to directly
  // match a const derived &, and so the compiler won't cast it up to
  // const irep_methods2 & and call the copy constructor. Fix this by
  // defining a copy constructor that exactly matches the (only) use case.
  irep_methods2(const derived &ref) : baseclass(ref)
  {
  }

  // The trait list is a std::tuple of field_traits<R, C, R C::*> entries.
  // for_each_field instantiates a default-constructed field_traits per
  // element (they carry only constexpr state) and lets the caller use
  // decltype(f) to recover R / C / value.
  template <typename F>
  static void for_each_field(F &&f)
  {
    std::apply(
      [&](auto... entries) { (f(entries), ...); }, typename traits::fields{});
  }

  // Method bodies live in irep2_meta_templates.h, which is included
  // after irep2_template_utils.h so the per-field-type helpers
  // (do_type_cmp, do_type_lt, do_type_crc, do_type_hash, do_type2string,
  // do_get_sub_expr*, call_expr_delegate, call_type_delegate) are
  // visible at the point of instantiation. Keeping the bodies out of
  // irep2.h avoids the dependency cycle that would otherwise force
  // irep2.h to include irep2_template_utils.h before the types those
  // helpers operate on (sideeffect_data::allockind etc.) are defined.
  base_container2tc clone() const override;
  list_of_memberst tostring(unsigned int indent) const override;
  bool cmp(const base2t &ref) const override;
  int lt(const base2t &ref) const override;
  size_t do_crc() const override;
  void hash(crypto_hash &h) const override;

protected:
  // Used by type_methods2 for subtype iteration. Definitions in
  // irep2_meta_templates.h.
  template <typename Delegate>
  void foreach_subtype_impl_const_inner(Delegate &f) const;
  template <typename Delegate>
  void foreach_subtype_impl_inner(Delegate &f);
};

/** Type methods template for type ireps.
 *  Like @expr_methods2, but for types. */
template <class derived, class baseclass, typename traits>
class type_methods2 : public irep_methods2<derived, baseclass, traits>
{
public:
  typedef irep_methods2<derived, baseclass, traits> superclass;

  template <typename... Args>
  type_methods2(const Args &...args) : superclass(args...)
  {
  }

  // See notes on irep_methods2 copy constructor
  type_methods2(const derived &ref) : superclass(ref)
  {
  }

  void
  foreach_subtype_impl_const(type2t::const_subtype_delegate &f) const override
  {
    this->foreach_subtype_impl_const_inner(f);
  }

  void foreach_subtype_impl(type2t::subtype_delegate &f) override
  {
    this->foreach_subtype_impl_inner(f);
  }
};

} // namespace esbmct

inline std::ostream &operator<<(std::ostream &out, const expr2tc &a)
{
  out << a->pretty(0);
  return out;
}

struct irep2_hash
{
  size_t operator()(const expr2tc &ref) const
  {
    return ref.crc();
  }
};

struct type2_hash
{
  size_t operator()(const type2tc &ref) const
  {
    return ref->crc();
  }
};

// Exception thrown by irep2_checked_type_cast / irep2_checked_expr_cast when
// the runtime id does not match the requested derived type. A bad to_*2t /
// to_*_type was always a logic bug — surfacing it as an exception keeps the
// failure deterministic in every build mode (rather than the undefined
// behaviour the previous NDEBUG dynamic_cast→static_cast redefine produced)
// while staying recoverable for callers that want to handle it.
class irep2_cast_error : public std::logic_error
{
public:
  using std::logic_error::logic_error;
};

// Diagnose a bad to_* downcast and throw irep2_cast_error. Out-of-line and
// [[noreturn]] so the happy path stays a single compare+branch and the
// compiler does not pessimise the inlined helpers.
[[noreturn]] void
irep2_bad_type_cast(unsigned actual, unsigned expected, const char *target);
[[noreturn]] void
irep2_bad_expr_cast(unsigned actual, unsigned expected, const char *target);
[[noreturn]] void irep2_bad_family_cast(unsigned actual, const char *accessor);

// Checked downcast for type2t / expr2t hierarchies. The is_*_type / is_*2t
// predicates already do a single enum compare; these helpers do the same
// check before a static_cast, so a bad to_*2t throws irep2_cast_error in
// every build mode rather than invoking undefined behaviour under NDEBUG.
template <typename Derived>
inline Derived &irep2_checked_type_cast(
  type2t &t,
  type2t::type_ids expected,
  const char *target_name)
{
  if (t.type_id != expected)
    irep2_bad_type_cast(t.type_id, expected, target_name);
  return static_cast<Derived &>(t);
}

template <typename Derived>
inline const Derived &irep2_checked_type_cast(
  const type2t &t,
  type2t::type_ids expected,
  const char *target_name)
{
  if (t.type_id != expected)
    irep2_bad_type_cast(t.type_id, expected, target_name);
  return static_cast<const Derived &>(t);
}

template <typename Derived>
inline Derived &irep2_checked_expr_cast(
  expr2t &e,
  expr2t::expr_ids expected,
  const char *target_name)
{
  if (e.expr_id != expected)
    irep2_bad_expr_cast(e.expr_id, expected, target_name);
  return static_cast<Derived &>(e);
}

template <typename Derived>
inline const Derived &irep2_checked_expr_cast(
  const expr2t &e,
  expr2t::expr_ids expected,
  const char *target_name)
{
  if (e.expr_id != expected)
    irep2_bad_expr_cast(e.expr_id, expected, target_name);
  return static_cast<const Derived &>(e);
}

#endif /* IREP2_H_ */
