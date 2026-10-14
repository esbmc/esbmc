#ifndef STRING_POOL_H
#define STRING_POOL_H

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>

class string_view_hash
{
public:
  using is_transparent = void;

  size_t operator()(std::string_view sv) const
  {
    return std::hash<std::string_view>{}(sv);
  }
};

class string_view_equal
{
public:
  using is_transparent = void;

  bool operator()(std::string_view a, std::string_view b) const
  {
    return a == b;
  }
};

// Append-only interning pool with lock-free reads.
//
// Storage is a segmented arena: a fixed-size table of chunk pointers, each
// chunk a contiguous array of std::string. Chunks are heap-allocated lazily
// under the writer mutex and published atomically. Once an index has been
// returned from get(), the slot at that index is immutable; readers can
// therefore look it up without any locking. The writer mutex only guards the
// hash table and chunk allocation.
class string_pool
{
public:
  template <typename T>
  unsigned operator[](const T &s)
  {
    return get(std::string_view(s));
  }

  string_pool()
  {
    // allocate empty string -- this gets index 0
    get(std::string_view(""));
  }
  ~string_pool() = default;

  // the pointer is guaranteed to be stable
  const char *c_str(size_t no) const
  {
    return get_string(no).c_str();
  }

  // the reference is guaranteed to be stable
  const std::string &get_string(size_t no) const
  {
    assert(no < published_size.load(std::memory_order_acquire));
    chunk_t *chunk = chunks[no >> CHUNK_SHIFT].load(std::memory_order_acquire);
    return chunk->data[no & CHUNK_MASK];
  }

protected:
  static constexpr size_t CHUNK_SHIFT = 12;
  static constexpr size_t CHUNK_SIZE = size_t{1} << CHUNK_SHIFT; // 4096
  static constexpr size_t CHUNK_MASK = CHUNK_SIZE - 1;
  // 2048 chunks * 4096 strings = 8M unique strings ceiling.
  static constexpr size_t MAX_CHUNKS = 2048;

  struct chunk_t
  {
    std::string data[CHUNK_SIZE];
  };

  typedef std::unordered_map<
    std::string_view,
    unsigned,
    string_view_hash,
    string_view_equal>
    hash_tablet;

  // Writer-side state, guarded by pool_mutex.
  hash_tablet hash_table;
  std::array<std::unique_ptr<chunk_t>, MAX_CHUNKS> chunk_storage{};

  // Reader-visible state. chunks[i] is published with release before
  // published_size is incremented past any index belonging to that chunk.
  std::array<std::atomic<chunk_t *>, MAX_CHUNKS> chunks{};
  std::atomic<size_t> published_size{0};

  mutable std::shared_mutex pool_mutex;

  unsigned get(std::string_view s);
};

inline string_pool &get_string_pool()
{
  static string_pool ret;
  return ret;
}

#endif
