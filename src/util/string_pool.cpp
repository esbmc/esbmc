#include <util/string_pool.h>

#include <cassert>
#include <mutex>
#include <shared_mutex>

unsigned string_pool::get(std::string_view s)
{
  {
    std::shared_lock lock(pool_mutex);
    hash_tablet::iterator it = hash_table.find(s);

    if (it != hash_table.end())
      return it->second;
  }

  std::unique_lock lock(pool_mutex);
  // Recheck after acquiring sole lock.
  hash_tablet::iterator it = hash_table.find(s);
  if (it != hash_table.end())
    return it->second;

  size_t no = hash_table.size();
  size_t chunk_idx = no >> CHUNK_SHIFT;
  size_t slot_idx = no & CHUNK_MASK;
  assert(chunk_idx < MAX_CHUNKS);

  if (slot_idx == 0)
  {
    // First slot of a chunk -- allocate and publish it before storing the
    // string. Readers that load this slot through chunks[chunk_idx] with
    // acquire will see a fully-constructed chunk_t.
    chunk_storage[chunk_idx] = std::make_unique<chunk_t>();
    chunks[chunk_idx].store(
      chunk_storage[chunk_idx].get(), std::memory_order_release);
  }

  chunk_t *chunk = chunk_storage[chunk_idx].get();
  chunk->data[slot_idx].assign(s.data(), s.size());

  // Key the hash table with a view into the now-stable storage.
  hash_table.emplace(std::string_view(chunk->data[slot_idx]), no);

  // Publish the new size last. A reader doing an acquire load of
  // published_size that observes a value > no will, by the release-acquire
  // edge, also observe the chunk pointer and the constructed string.
  published_size.store(no + 1, std::memory_order_release);

  return static_cast<unsigned>(no);
}
