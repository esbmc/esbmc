#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>

typedef struct
{
  const void *value; // data pointer
  size_t type_id;    // hashed type name
  size_t size;       // number of bytes in value
} Object;

typedef struct
{
  Object *items;
  size_t size; // elements in use
} List;

/* ---------- create ---------- */
static inline List *list_create(Object *backing)
{
  List *l = __ESBMC_alloca(sizeof(List));

  __ESBMC_assume(l != NULL);

  l->items = backing;
  l->size = 0;
  return l;
}

/* ---------- bounds check ---------- */
/*
static inline bool list_in_bounds(const List *l, size_t index)
{
  return index < l->size;
}
*/

static bool list_eq(const List *l1, const List *l2)
{
  if (!l1 || !l2)
    return false;
  if (__ESBMC_same_object(l1, l2))
    return true;
  if (l1->size != l2->size)
    return false;

  size_t i = 0;
  while (i < l1->size)
  {
    const Object *a = &l1->items[i];
    const Object *b = &l2->items[i];

    // Same address => element equal; keep checking the rest.
    if (a->value == b->value)
    {
      ++i;
      continue;
    }

    if (
      !a->value || !b->value || a->type_id != b->type_id ||
      a->size != b->size || memcmp(a->value, b->value, a->size) != 0)
    {
      return false;
    }

    ++i;
  }
  return true;
}

static inline size_t list_size(const List *l)
{
  return l ? l->size : 0;
}

/* ---------- getters ---------- */
static inline Object *list_at(List *l, size_t index)
{
  //return list_in_bounds(l, index) ? &l->items[index] : NULL;
  assert(index < l->size);
  return &l->items[index];
}

static inline const Object *list_cat(const List *l, size_t index)
{
  //  return list_in_bounds(l, index) ? &l->items[index] : NULL;
  assert(index < l->size);
  return &l->items[index];
}

static inline void *list_get_as(const List *l, size_t i, size_t expect_type)
{
  const Object *o = list_cat(l, i);
  return (o && o->type_id == expect_type) ? (void *)o->value : NULL;
}

/* ---------- push element ---------- */

/*
static inline bool list_push(List *l, const void *value, size_t type_id)
{
  l->items[l->size].value = value;
  l->items[l->size].type_id = type_id;
  l->size++;
  return true;
}
*/

static inline bool
list_push(List *l, const void *value, size_t type_id, size_t type_size)
{
  void *copied_value = __ESBMC_alloca(type_size);

  memcpy(copied_value, value, type_size);

  l->items[l->size].value = copied_value;
  l->items[l->size].type_id = type_id;
  l->items[l->size].size = type_size;
  l->size++;
  return true;
}

static inline bool list_push_object(List *l, Object *o)
{
  assert(l != NULL);
  assert(o != NULL);
  return list_push(l, o->value, o->type_id, o->size);
}

/* ---------- insert element at index ---------- */
static inline bool list_insert(
  List *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size)
{
  // If index is beyond the end, just append
  if (index >= l->size)
    return list_push(l, value, type_id, type_size);

  // Make a copy of the value
  void *copied_value = __ESBMC_alloca(type_size);
  memcpy(copied_value, value, type_size);

  // Shift all elements from index onwards one position to the right
  size_t elements_to_shift = l->size - index;
  memmove(
    &l->items[index + 1], &l->items[index], elements_to_shift * sizeof(Object));

  // Insert the new element
  l->items[index].value = copied_value;
  l->items[index].type_id = type_id;
  l->items[index].size = type_size;
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

static bool list_contains(
  const List *l,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  if (!l || !item)
    return false;

  size_t i = 0;
  while (i < l->size)
  {
    const Object *elem = &l->items[i];

    // Check if types and sizes match
    if (elem->type_id == item_type_id && elem->size == item_size)
    {
      // Compare the actual data
      if (elem->value == item || memcmp(elem->value, item, item_size) == 0)
        return true;
    }

    ++i;
  }
  return false;
}

/* ---------- extend list ---------- */
static inline void list_extend(List *l, const List *other)
{
  if (!l || !other)
    return;

  size_t i = 0;
  while (i < other->size)
  {
    const Object *elem = &other->items[i];

    void *copied_value = __ESBMC_alloca(elem->size);
    memcpy(copied_value, elem->value, elem->size);

    l->items[l->size].value = copied_value;
    l->items[l->size].type_id = elem->type_id;
    l->items[l->size].size = elem->size;
    l->size++;

    ++i;
  }
}
