#ifndef GOTO_SYMEX_LEVEL1_MAP_H_
#define GOTO_SYMEX_LEVEL1_MAP_H_

#include <cstddef>
#include <immer/map.hpp>
#include <immer/map_transient.hpp>
#include <immer/memory_policy.hpp>

/** Persistent L1-renaming map with O(1) copy.
 *
 *  level1t::current_names maps an L1 name_record to its frame number. It
 *  lives inside framet, and framet is deep-copied whenever the symex call
 *  stack is snapshotted (every per-branch merge_statet, every state fork).
 *  With a std::unordered_map that copy is O(N) buckets + N entries; backing
 *  it with immer::map (a HAMT) makes the copy O(1) structural sharing, the
 *  same win guard_seq's immer::vector gives the conjunct list.
 *
 *  Ownership: non-atomic refcount, no lock — guards/renaming run in the
 *  single-threaded symex phase (the only parallelism, --parallel-solving,
 *  solves an already-built equation and never touches a live framet). See
 *  guard_seq.h for the same reasoning.
 *
 *  The exposed surface is just what level1t uses: set / erase / lookup /
 *  iterate. Lookup returns `const V*` (immer's find), nullptr when absent —
 *  a pointer into shared storage, valid until the next mutating call. */
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

  // nullptr if absent; otherwise a pointer into shared storage valid until
  // the next mutation. Callers must not stash it across a set()/erase().
  const V *find(const K &k) const
  {
    return m_.find(k);
  }

  // Insert or overwrite. O(log N), shares untouched subtrees.
  void set(const K &k, const V &v)
  {
    m_ = std::move(m_).set(k, v);
  }

  void erase(const K &k)
  {
    m_ = std::move(m_).erase(k);
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
};

#endif /* GOTO_SYMEX_LEVEL1_MAP_H_ */
