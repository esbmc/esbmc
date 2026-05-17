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

namespace esbmct
{

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
private:
  // Private tag gate for the raw-pointer adopt constructor. Only
  // `make_irep` and other befriended factories can construct one,
  // which means user code cannot do `expr2tc(new foo2t(...))` — the
  // canonical idiom is the per-kind `foo2tc(...)` factory (itself a
  // thin wrapper over make_irep). The tag carries no state.
  struct make_tag
  {
    explicit make_tag() = default;
  };

  // Adopt a freshly-allocated node. The pointee's refcount must be 0
  // (i.e. just-new'd); we bump it to 1.
  irep_container(T *raw, make_tag) noexcept : ptr_(raw)
  {
    if (ptr_)
      ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
  }

  template <class U, class... Args>
  friend irep_container<typename U::base_type> make_irep(Args &&...args);

public:
  // Default: empty container, nullptr inside. `noexcept`/`constexpr` to
  // preserve the storage-class properties of the previous design.
  constexpr irep_container() noexcept : ptr_(nullptr)
  {
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
    // synchronises with the writer that produced it (crc()'s release
    // store), and we observe whatever node state went into computing it.
    if (size_t cached = foo->crc_val.load(std::memory_order_acquire);
        cached != 0)
      return cached;
    return foo->crc();
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
  using container_t = irep_container<typename T::base_type>;
  return container_t(
    new T(std::forward<Args>(args)...), typename container_t::make_tag{});
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

  void foreach_subtype_impl_const(const_subtype_delegate &t) const;
  void foreach_subtype_impl(subtype_delegate &t);

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
  unsigned int get_width() const;

  bool operator==(const type2t &ref) const;
  bool operator!=(const type2t &ref) const;
  bool operator<(const type2t &ref) const;

  /** Produce a human-readable string representation of type.
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

  /** Structural comparison. Caller-side contract: @p ref's `type_id`
   *  must already match this one's (the kind dispatch happens by
   *  switch on type_id internally). Use cmpchecked when that hasn't
   *  been verified upstream.
   *  @see cmpchecked
   *  @param ref Reference to (same kind of) type to compare against
   *  @return True if types match, false otherwise
   */
  bool cmp(const type2t &ref) const;

  /** Trinary structural ordering. Switch-dispatches on type_id then
   *  walks the kind's fields, mirroring `cmp` but returning -1/0/+1.
   *  Use ltchecked when @p ref's kind hasn't been verified upstream.
   *  @see ltchecked
   *  @param ref Reference to (same kind of) type to measure against
   *  @return 0 if types are the same, 1 if this > ref, -1 if ref > this.
   */
  int lt(const type2t &ref) const;

  /** Extract a list of members from type as strings.
   *  Produces a list of pairs, mapping a member name to a string value. Used
   *  in the body of the pretty method.
   *  @see pretty
   *  @param indent Number of spaces to indent output strings with, if multiline
   *  @return list of name:value pairs.
   */
  list_of_memberst tostring(unsigned int indent) const;

  /** Perform hash operation accumulating into parameter.
   *  Feeds data as appropriate to the type of the expression into the
   *  parameter, to be hashed. Like crc, but for some other kind of hash
   *  scenario.
   *  @see cmp
   *  @see crc
   *  @param hash Object to accumulate hash data into.
   */
  void hash(crypto_hash &hash) const;

  /** Clone method. Self explanatory.
   *  @return New container, containing a duplicate of this object.
   */
  type2tc clone() const;

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

  /** Instance of type_ids recording this types type. */
  const type_ids type_id;

  // CRC cache: 0 means "not yet computed". Atomic so concurrent readers
  // see either the prior value or the fresh one — never a torn read.
  // Writers respect the single-writer contract documented at the top of
  // this header; the cache is therefore safe to set without a lock as
  // long as readers and the (single) writer share happens-before via the
  // atomic.
  mutable std::atomic<size_t> crc_val;
};

/** Fetch identifying name for a type.
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

  void foreach_operand_impl_const(const_op_delegate &expr) const;
  void foreach_operand_impl(op_delegate &expr);

public:
  // Provide base / container types for some templates stuck on top:
  typedef expr2tc container_type;
  typedef expr2t base_type;

  virtual ~expr2t() = default;

  expr2tc clone() const;

  /** Build a fresh expression of the same kind and field values as this
   *  one, but with a different type. Assumes every kind's primary
   *  constructor takes (type, fields...) matching its K::fields tuple
   *  in order. */
  expr2tc with_type(const type2tc &new_type) const;

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

  /** Structural comparison. The expr_id check is done by the switch
   *  dispatcher inside, so callers don't have to gate the call on kind.
   *  Should normally be reached via operator==.
   *  @see type2t::cmp
   *  @param ref Expr object to compare this against
   *  @return True if objects are the same; false otherwise.
   */
  bool cmp(const expr2t &ref) const;

  /** Trinary structural ordering. Mirrors `cmp` but returns -1/0/+1.
   *  Like `cmp`, the expr_id check is internal to the switch
   *  dispatcher; callers don't need to gate on kind. Normally reached
   *  via operator< or ltchecked.
   *  @see type2t::lt
   *  @param ref Expr object to compare this against
   *  @return 0 If exprs are the same, 1 if this > ref, -1 if ref > this.
   */
  int lt(const expr2t &ref) const;

  /** Convert fields of subclasses to a string representation.
   *  Used internally by the pretty method - creates a list of pairs
   *  representing the fields in the subclass. Each pair is a pair of strings
   *  of the form fieldname : value. The value may be multiline, in which case
   *  the new line will have at least indent number of indenting spaces.
   *  @param indent Number of spaces to indent multiline output by
   *  @return list of string pairs, of form fieldname:value
   */
  list_of_memberst tostring(unsigned int indent) const;

  /** Perform hash operation accumulating into parameter.
   *  Feeds data as appropriate to the type of the expression into the
   *  parameter, to be hashed. Like crc, but for some other kind of hash
   *  scenario.
   *  @see cmp
   *  @see crc
   *  @param hash Object to accumulate hash data into.
   */
  void hash(crypto_hash &hash) const;

  /** Fetch a sub-operand.
   *  These can come out of any field that is an expr2tc, or contains them.
   *  No particular numbering order is promised.
   */
  const expr2tc *get_sub_expr(size_t idx) const;

  /** Count the number of sub-exprs there are.
   */
  size_t get_num_sub_exprs() const;

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
    foreach_operand_impl_const(wrapped);
  }

  template <typename T>
  void Foreach_operand(T &&t)
  {
    op_delegate wrapped(t);
    foreach_operand_impl(wrapped);
  }

  /** Instance of expr_ids recording tihs exprs type. */
  const expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  const type2tc type;

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

/** Containers and field-walking infrastructure for irep2 nodes.
 *
 *  Each concrete kind (e.g. `add2t`, `if2t`, `array_type2t`) inherits
 *  directly from `expr2t` / `type2t` and carries:
 *
 *    static constexpr auto fields = std::make_tuple(&Kind::field1, ...);
 *    static std::string field_names[num_type_fields];
 *
 *  The generic_*<K> helpers in `irep2_dispatch.h` walk that tuple via
 *  std::apply to implement clone/cmp/lt/crc/hash/tostring and operand
 *  iteration. The switch-on-id dispatchers on `expr2t` / `type2t`
 *  select the right K per kind from the X-macro manifests
 *  (`expr_kinds.inc`, `type_kinds.inc`).
 *
 *  Each kind also has a container alias `<kind>2tc` (alias for
 *  `irep_container<expr2t>` / `irep_container<type2t>`) and a factory
 *  `<kind>2tc(args...)` that wraps `make_irep<T>(args...)`. Containers
 *  behave like value types via copy-on-write: const access shares the
 *  pointee, non-const `get()` / `operator->` / `operator*` calls
 *  `detach()` first (clones into a fresh refcount=1 object when shared).
 *
 *  Identification and downcasting helpers `is_<kind>2t(expr)`,
 *  `to_<kind>2t(expr)`, `try_to_<kind>2t(expr)` are macro-generated by
 *  `irep2_expr.h` / `irep2_type.h` from the same manifests. `to_*`
 *  throws `irep2_cast_error` on a kind mismatch; `try_to_*` returns
 *  `nullptr`. The non-const downcast goes through the non-const
 *  container accessor, so it triggers detach automatically.
 */
namespace esbmct
{
/** Maximum number of fields to support in expr2t subclasses. This value
 *  controls the types of any arrays that need to consider the number of
 *  fields.
 *  I've yet to find a way of making this play nice with the new variardic
 *  way of defining ireps. */
const unsigned int num_type_fields = 6;

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
