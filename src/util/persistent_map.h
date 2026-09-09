#ifndef UTIL_PERSISTENT_MAP_H_
#define UTIL_PERSISTENT_MAP_H_

#include <cstddef>
#include <functional>
#include <immer/algorithm.hpp>
#include <immer/map.hpp>
#include <immer/map_transient.hpp>
#include <immer/memory_policy.hpp>
#include <immer/set.hpp>

/** Persistent (structurally shared) map with O(1) copy.
 *
 *  Symex state that is snapshotted whenever the call stack is copied — the
 *  level1/level2 renaming maps and the value set — lives in one of these.
 *  A std::unordered_map copies O(N) at every per-branch merge_statet and
 *  state fork; immer's HAMT makes that copy O(1) structural sharing, the
 *  same win guard_seq's immer::vector gives the conjunct list. Two shared
 *  snapshots then compare in O(divergence) via diff(), which is what turns
 *  a control-flow merge from O(state) into O(paths' divergence).
 *
 *  Ownership: non-atomic refcount, no lock — symex runs single-threaded (the
 *  only parallelism, --parallel-solving, solves an already-built equation and
 *  never touches a live state). See guard_seq.h for the same reasoning.
 *
 *  Lookup returns `const V*` (immer's find), nullptr when absent — a pointer
 *  into shared storage, valid until the next mutating call; callers must not
 *  stash it across a set()/erase(). */
template <typename K, typename V, typename Hash>
class persistent_map
{
  using memory_policy = immer::memory_policy<
    immer::default_heap_policy,
    immer::unsafe_refcount_policy,
    immer::no_lock_policy>;
  using map_t = immer::map<K, V, Hash, std::equal_to<K>, memory_policy>;

  map_t m_;

public:
  persistent_map() = default;

  std::size_t size() const
  {
    return m_.size();
  }

  bool empty() const
  {
    return m_.size() == 0;
  }

  // nullptr if absent; otherwise a pointer into shared storage valid until
  // the next mutation. Callers must not stash it across a set()/erase().
  const V *find(const K &k) const
  {
    return m_.find(k);
  }

  // The value at a key known to be present. Like find(), the reference is
  // valid only until the next mutation.
  const V &at(const K &k) const
  {
    return m_.at(k);
  }

  // Insert or overwrite. O(log N), shares untouched subtrees.
  void set(const K &k, const V &v)
  {
    m_ = std::move(m_).set(k, v);
  }

  // Move-in overload: immer's set takes its value by value, so handing it an
  // rvalue moves rather than copies.
  void set(const K &k, V &&v)
  {
    m_ = std::move(m_).set(k, std::move(v));
  }

  // Read-modify-write in one HAMT walk instead of find + set. fn receives the
  // current value (value-initialised when the key is absent) and returns the
  // new one.
  template <class Fn>
  void update(const K &k, Fn &&fn)
  {
    m_ = std::move(m_).update(k, std::forward<Fn>(fn));
  }

  // Erase; true when the key was present. O(log N), shares subtrees.
  bool erase(const K &k)
  {
    std::size_t n = m_.size();
    m_ = std::move(m_).erase(k);
    return m_.size() != n;
  }

  void clear()
  {
    m_ = map_t{};
  }

  using const_iterator = typename map_t::iterator;
  const_iterator begin() const
  {
    return m_.begin();
  }
  const_iterator end() const
  {
    return m_.end();
  }

  // Structural diff against another map: added(kv) for keys in `other`
  // not here, removed(kv) for keys here not in `other`, changed(a, b)
  // for a shared key whose value differs. O(|diff|) when the two maps
  // share structure (both descend from a common snapshot), which is
  // the case at a control-flow merge — so a merge costs the paths'
  // divergence, not the whole map.
  template <class Added, class Removed, class Changed>
  void diff(
    const persistent_map &other,
    Added &&added,
    Removed &&removed,
    Changed &&changed) const
  {
    immer::diff(
      m_,
      other.m_,
      immer::make_differ(
        std::forward<Added>(added),
        std::forward<Removed>(removed),
        std::forward<Changed>(changed)));
  }
};

/** Persistent set with O(1) copy, the value-less analogue of persistent_map.
 *
 *  framet::local_variables and declaration_history hold the frame's L1 name
 *  records. local_variables is copied into every per-branch merge_statet and
 *  union-merged at each join (merge_locality); with a std::unordered_set that
 *  is O(N) per branch, so a method holding N locals across N joins is O(N^2).
 *  Backing it with immer::set makes the snapshot O(1) and the union O(|diff|)
 *  — see the diff() note below and persistent_map's header. */
template <typename K, typename Hash>
class persistent_set
{
  using memory_policy = immer::memory_policy<
    immer::default_heap_policy,
    immer::unsafe_refcount_policy,
    immer::no_lock_policy>;
  using set_t = immer::set<K, Hash, std::equal_to<K>, memory_policy>;

  set_t s_;

public:
  persistent_set() = default;

  std::size_t size() const
  {
    return s_.size();
  }

  bool empty() const
  {
    return s_.size() == 0;
  }

  // 1 if present, 0 otherwise.
  std::size_t count(const K &k) const
  {
    return s_.count(k);
  }

  // Insert. O(log N), shares untouched subtrees.
  void insert(const K &k)
  {
    s_ = std::move(s_).insert(k);
  }

  // Erase; true when the key was present. O(log N), shares subtrees.
  bool erase(const K &k)
  {
    std::size_t n = s_.size();
    s_ = std::move(s_).erase(k);
    return s_.size() != n;
  }

  using const_iterator = typename set_t::iterator;
  const_iterator begin() const
  {
    return s_.begin();
  }
  const_iterator end() const
  {
    return s_.end();
  }

  // Structural diff against another set: added(k) for a key in `other` not
  // here, removed(k) for one here not in `other`. O(|diff|) when the two
  // sets share structure (both descend from a common snapshot) — so the
  // union at a control-flow merge costs the paths' divergence, not N.
  template <class Added, class Removed>
  void diff(const persistent_set &other, Added &&added, Removed &&removed) const
  {
    immer::diff(
      s_,
      other.s_,
      immer::make_differ(
        std::forward<Added>(added), std::forward<Removed>(removed)));
  }
};

#endif /* UTIL_PERSISTENT_MAP_H_ */
