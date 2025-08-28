#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>


typedef struct
{
  const void *value; // data pointer
  size_t type_id;    // hashed type name
} Object;

typedef struct
{
  Object *items;
  size_t size; // elements in use
} List;

/* ---------- init ---------- */
static inline bool list_init(List *l, Object *backing)
{
  l->items = backing;
  l->size = 0;
  return true;
}

/* ---------- bounds check ---------- */
static inline bool list_in_bounds(const List *l, size_t index)
{
  return index < l->size;
}

/* ---------- getters ---------- */
static inline Object *list_at(List *l, size_t index)
{
  return list_in_bounds(l, index) ? &l->items[index] : NULL;
}

static inline const Object *list_cat(const List *l, size_t index)
{
  return list_in_bounds(l, index) ? &l->items[index] : NULL;
}

static inline void *list_get_as(const List *l, size_t i, size_t expect_type)
{
  const Object *o = list_cat(l, i);
  return (o && o->type_id == expect_type) ? (void *)o->value : NULL;
}

/* ---------- push element ---------- */
static inline bool list_push(List *l, const void *value, size_t type_id)
{
  l->items[l->size].value = value;
  l->items[l->size].type_id = type_id;
  l->size++;
  return true;
}

/* ---------- replace element ---------- */
static inline bool
list_replace(List *l, size_t index, const void *new_value, size_t type_id)
{
  if (index >= l->size)
    return false;
  l->items[index].value = new_value;
  l->items[index].type_id = type_id;
  return true;
}

/* ---------- pop / erase ---------- */
static inline bool list_pop(List *l)
{
  if (l->size == 0)
    return false;
  l->size--;
  return true;
}

/* ---------- type hashing ---------- */
static inline size_t list_hash_string(const char *str)
{
  size_t hash = 5381;
  int c;
  while ((c = *str++))
  {
    hash = ((hash << 5) + hash) + c;
  }
  return hash;
}


// Append N elements, keeping the buffer ALWAYS with the exact size.
// - array: current pointer (can be NULL when len == 0)
// - len:   current number of elements
// - elem_size: size of each element in bytes
// - src:   pointer to N contiguous elements to append
// - n:     number of elements to append
// Returns a new pointer on success (old block is freed);
// NULL on failure (old array is left untouched).
/*
void* __list_append__(
  void *array, size_t len, size_t elem_size, const void *src, size_t n)
{
  if (n == 0)
    return array;

  if (!src || elem_size == 0)
    return NULL;

  // Check overflow in len + n
  if (len > SIZE_MAX - n)
    return NULL;

  size_t new_len = len + n;

  // Check overflow in new_len * elem_size
  if (elem_size != 0 && new_len > SIZE_MAX / elem_size)
    return NULL;

  size_t old_bytes = len * elem_size;
  size_t new_bytes = new_len * elem_size;

  assert(new_bytes > 0);

  // Allocate new block with the required size
  void *newbuf = malloc(new_bytes);
  if (!newbuf) {
    assert(0);
    return NULL;
  }

  // // Copy old elements if any
  if (array && old_bytes)
  {
    memcpy(newbuf, array, old_bytes);
  }

  // // Copy the new elements at the end
  memcpy((char *)newbuf + old_bytes, src, n * elem_size);

  // Free the old block
  if (array)
    free(array);

  return newbuf;
}
*/
