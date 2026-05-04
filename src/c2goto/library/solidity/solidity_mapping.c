/* Solidity mapping data structure */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "solidity_types.h"

struct _ESBMC_Mapping
{
  address_t addr : 160;
  uint256_t key : 256;
  void *value;
  struct _ESBMC_Mapping *next;
} __attribute__((packed));

struct mapping_t
{
  struct _ESBMC_Mapping *base;
  address_t addr : 160;
} __attribute__((packed));

void *map_get_raw(struct _ESBMC_Mapping a[], address_t addr, uint256_t key)
{
__ESBMC_HIDE:;
  struct _ESBMC_Mapping *cur = a[0].next;
  while (cur)
  {
    if (cur->addr == addr && cur->key == key)
      return cur->value;
    cur = cur->next;
  }
  return NULL;
}

void map_set_raw(struct _ESBMC_Mapping a[], address_t addr,
                 uint256_t key, void *val)
{
__ESBMC_HIDE:;
  struct _ESBMC_Mapping *n = (struct _ESBMC_Mapping *)malloc(sizeof *n);
  n->addr = addr;
  n->key = key;
  n->value = val;
  n->next = a[0].next;
  a[0].next = n;
}

/* uint256_t */
void map_uint_set(struct mapping_t *m, uint256_t k, uint256_t v)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw(m->base, m->addr, k, p);
}
uint256_t map_uint_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)map_get_raw(m->base, m->addr, k);
  return p ? *p : (uint256_t)0;
}

/* int256_t */
void map_int_set(struct mapping_t *m, uint256_t k, int256_t v)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw(m->base, m->addr, k, p);
}
int256_t map_int_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)map_get_raw(m->base, m->addr, k);
  return p ? *p : (int256_t)0;
}

/* string */
void map_string_set(struct mapping_t *m, uint256_t k, char *v)
{
__ESBMC_HIDE:;
  char **p = (char **)malloc(sizeof *p);
  *p = v;
  map_set_raw(m->base, m->addr, k, p);
}
char *map_string_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  char **p = (char **)map_get_raw(m->base, m->addr, k);
  return p ? *p : (char *)0;
}

/* bool */
void map_bool_set(struct mapping_t *m, uint256_t k, bool v)
{
__ESBMC_HIDE:;
  bool *p = (bool *)malloc(sizeof *p);
  *p = v;
  map_set_raw(m->base, m->addr, k, p);
}

bool map_bool_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  bool *p = (bool *)map_get_raw(m->base, m->addr, k);
  return p ? *p : false;
}

/* generic */
void map_generic_set(struct mapping_t *m, uint256_t k, const void *v, size_t sz)
{
__ESBMC_HIDE:;
  void *p = malloc(sz);
  memcpy(p, v, sz);
  map_set_raw(m->base, m->addr, k, p);
}
void *map_generic_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  return map_get_raw(m->base, m->addr, k);
}

struct _ESBMC_Mapping_fast
{
  uint256_t key : 256;
  void *value;
  struct _ESBMC_Mapping_fast *next;
} __attribute__((packed));

struct mapping_t_fast
{
  struct _ESBMC_Mapping_fast *base;
};

void *map_get_raw_fast(struct _ESBMC_Mapping_fast a[], uint256_t key)
{
__ESBMC_HIDE:;
  struct _ESBMC_Mapping_fast *cur = a[0].next;
  while (cur)
  {
    if (cur->key == key)
      return cur->value;
    cur = cur->next;
  }
  return NULL;
}

void map_set_raw_fast(struct _ESBMC_Mapping_fast a[],
                      uint256_t key, void *val)
{
__ESBMC_HIDE:;
  struct _ESBMC_Mapping_fast *n = (struct _ESBMC_Mapping_fast *)malloc(sizeof *n);
  n->key = key;
  n->value = val;
  n->next = a[0].next;
  a[0].next = n;
}

/* uint256_t */
void map_uint_set_fast(struct mapping_t_fast *m, uint256_t k, uint256_t v)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m->base, k, p);
}
uint256_t map_uint_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)map_get_raw_fast(m->base, k);
  return p ? *p : (uint256_t)0;
}

/* int256_t */
void map_int_set_fast(struct mapping_t_fast *m, uint256_t k, int256_t v)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m->base, k, p);
}
int256_t map_int_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)map_get_raw_fast(m->base, k);
  return p ? *p : (int256_t)0;
}

/* string */
void map_string_set_fast(struct mapping_t_fast *m, uint256_t k, char *v)
{
__ESBMC_HIDE:;
  char **p = (char **)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m->base, k, p);
}
char *map_string_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  char **p = (char **)map_get_raw_fast(m->base, k);
  return p ? *p : (char *)0;
}

/* bool */
void map_bool_set_fast(struct mapping_t_fast *m, uint256_t k, bool v)
{
__ESBMC_HIDE:;
  bool *p = (bool *)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m->base, k, p);
}
bool map_bool_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  bool *p = (bool *)map_get_raw_fast(m->base, k);
  return p ? *p : false;
}

/* generic */
void map_generic_set_fast(struct mapping_t_fast *m, uint256_t k, const void *v, size_t sz)
{
__ESBMC_HIDE:;
  void *p = malloc(sz);
  memcpy(p, v, sz);
  map_set_raw_fast(m->base, k, p);
}
void *map_generic_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  return map_get_raw_fast(m->base, k);
}
