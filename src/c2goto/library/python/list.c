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
  List *l = malloc(sizeof(List));

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
  assert(l1 && l2);
  if (l1->size != l2->size)
    return false;

  size_t i = 0;
  while (i < l1->size)
  {
    const Object *a = &l1->items[i];
    const Object *b = &l2->items[i];

    if (a->type_id != b->type_id)
      return false;
    if (a->size != b->size)
      return false;

    // Same address => element equal; keep checking the rest.
    if (a->value == b->value)
    {
      ++i;
      continue;
    }

    // If either is NULL (and not the same address), not equal.
    if (!a->value || !b->value)
      return false;

    // memcmp
    const unsigned char *pa = (const unsigned char *)a->value;
    const unsigned char *pb = (const unsigned char *)b->value;

    size_t j = 0;
    while (j < a->size)
    {
      if (pa[j] != pb[j])
        return false;
      ++j;
    }

    ++i;
  }
  return true;
}

static long long int list_size(const List *l)
{
  assert(l);
  return l->size;
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
  void *copied_value = malloc(type_size);

  // Force malloc to succeed for verification
  __ESBMC_assume(copied_value != NULL);

  if (copied_value == NULL)
  {
    return false;
  }

  // memcpy
  const char *src = (const char *)value;
  char *dst = (char *)copied_value;
  size_t i = 0;
  while (i < type_size)
  {
    dst[i] = src[i];
    i++;
  }

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
