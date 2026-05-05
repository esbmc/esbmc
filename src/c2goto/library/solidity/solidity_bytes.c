/* Solidity bytes type operations */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include "solidity_types.h"

typedef struct BytesPool
{
  unsigned char *pool;
  size_t pool_cursor;
} BytesPool;

typedef struct BytesStatic
{
  unsigned char data[32];
  size_t length;
} BytesStatic;

void bytes_dynamic_init_check(const int initialized)
{
__ESBMC_HIDE:;
  if (initialized == 0)
    assert(!"Uninitialized Dynamic Bytes");
}

void bytes_dynamic_bounds_check(size_t index, size_t length)
{
__ESBMC_HIDE:;
  if (index >= length)
    assert(!"Out-of-bounds access on Dynamic Bytes");
}

unsigned char hex_char_to_nibble(char c)
{
__ESBMC_HIDE:;
  if ('0' <= c && c <= '9')
    return c - '0';
  else if ('a' <= tolower(c) && tolower(c) <= 'f')
    return tolower(c) - 'a' + 10;
  else
    abort();
  return 0;
}

BytesStatic bytes_static_from_hex(const char *hex_str, size_t len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
  size_t hex_len = len - 2;
  b.length = hex_len / 2;
  for (size_t i = 0; i < b.length; i++)
  {
    unsigned char high = hex_char_to_nibble(hex_str[2 + i * 2]);
    unsigned char low = hex_char_to_nibble(hex_str[2 + i * 2 + 1]);
    b.data[i] = (high << 4) | low;
  }
  return b;
}

BytesStatic bytes_static_from_string(const char *str, size_t len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
  if (len > 32)
    len = 32;
  size_t copy_len = strnlen(str, len);
  memcpy(b.data, str, copy_len);
  memset(b.data + copy_len, 0, len - copy_len);
  b.length = len;
  return b;
}

BytesStatic bytes_static_truncate(const BytesStatic *src, size_t new_len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
  memcpy(b.data, src->data, new_len);
  b.length = new_len;
  return b;
}

BytesStatic bytes_static_and(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
  for (size_t i = 0; i < a->length; i++)
  {
    r.data[i] = a->data[i] & b->data[i];
  }
  r.length = a->length;
  return r;
}

BytesStatic bytes_static_or(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
  for (size_t i = 0; i < a->length; i++)
  {
    r.data[i] = a->data[i] | b->data[i];
  }
  r.length = a->length;
  return r;
}

BytesStatic bytes_static_xor(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
  for (size_t i = 0; i < a->length; i++)
  {
    r.data[i] = a->data[i] ^ b->data[i];
  }
  r.length = a->length;
  return r;
}

uint256_t bytes_static_to_uint(const BytesStatic *b)
{
__ESBMC_HIDE:;
  uint256_t result = 0;
  for (size_t i = 0; i < b->length; i++)
  {
    result = (result << 8) | b->data[i];
  }
  return result;
}

BytesStatic bytes_static_from_uint(uint256_t val, size_t len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
  for (size_t i = 0; i < len; i++)
  {
    b.data[len - 1 - i] = val & 0xFF;
    val >>= 8;
  }
  b.length = len;
  return b;
}

BytesStatic bytes_static_shl(const BytesStatic *src, unsigned shift_bits)
{
__ESBMC_HIDE:;
  uint256_t val = bytes_static_to_uint(src);
  val <<= shift_bits;
  return bytes_static_from_uint(val, src->length);
}

BytesStatic bytes_static_shr(const BytesStatic *src, unsigned shift_bits)
{
__ESBMC_HIDE:;
  uint256_t val = bytes_static_to_uint(src);
  val >>= shift_bits;
  return bytes_static_from_uint(val, src->length);
}

uint256_t bytes_static_to_mapping_key(const BytesStatic *b)
{
__ESBMC_HIDE:;
  return ((uint256_t)b->length << 248) | bytes_static_to_uint(b);
}

BytesStatic bytes_static_init_zero(size_t len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
  b.length = len;
  memset(b.data, 0, len);
  return b;
}

BytesDynamic bytes_dynamic_init_zero(size_t len, BytesPool *pool)
{
__ESBMC_HIDE:;
  BytesDynamic b = {0};
  b.offset = pool->pool_cursor;
  b.length = len;
  b.capacity = len;
  b.initialized = 1;
  memset(&pool->pool[b.offset], 0, len);
  pool->pool_cursor += len;
  return b;
}

void bytes_dynamic_init(
  BytesDynamic *b,
  const unsigned char *input,
  size_t len,
  BytesPool *pool)
{
__ESBMC_HIDE:;
  b->offset = pool->pool_cursor;
  b->length = len;
  b->capacity = len;
  b->initialized = 1;
  memcpy(&pool->pool[b->offset], input, len);
  pool->pool_cursor += len;
}

void bytes_dynamic_ensure_capacity(
  BytesDynamic *b,
  size_t required,
  BytesPool *pool)
{
__ESBMC_HIDE:;
  if (required <= b->capacity)
    return;
  size_t new_capacity = b->capacity;
  if (new_capacity == 0)
    new_capacity = 1;
  while (new_capacity < required)
    new_capacity *= 2;
  size_t new_offset = pool->pool_cursor;
  memcpy(&pool->pool[new_offset], &pool->pool[b->offset], b->length);
  b->offset = new_offset;
  b->capacity = new_capacity;
  pool->pool_cursor += new_capacity;
}

BytesDynamic bytes_dynamic_from_static(const BytesStatic *s, BytesPool *pool)
{
__ESBMC_HIDE:;
  BytesDynamic b = {0};
  bytes_dynamic_init(&b, s->data, s->length, pool);
  return b;
}

BytesDynamic bytes_dynamic_from_string(const char *str, BytesPool *pool)
{
__ESBMC_HIDE:;
  BytesDynamic b = {0};
  bytes_dynamic_init(
    &b, (const unsigned char *)str, strnlen(str, SIZE_MAX - 1), pool);
  return b;
}

BytesDynamic
bytes_dynamic_from_hex(const char *hex_str, size_t len, BytesPool *pool)
{
__ESBMC_HIDE:;
  size_t hex_len = len - 2;
  size_t byte_len = hex_len / 2;
  unsigned char tmp[32] = {0};
  for (size_t i = 0; i < byte_len; i++)
  {
    unsigned char high = hex_char_to_nibble(hex_str[2 + i * 2]);
    unsigned char low = hex_char_to_nibble(hex_str[2 + i * 2 + 1]);
    tmp[i] = (high << 4) | low;
  }
  BytesDynamic b = {0};
  bytes_dynamic_init(&b, tmp, byte_len, pool);
  return b;
}

BytesStatic bytes_static_truncate_from_dynamic(
  const BytesDynamic *src,
  size_t new_len,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  BytesStatic b = {0};
  memcpy(b.data, &pool->pool[src->offset], new_len);
  b.length = new_len;
  return b;
}

BytesDynamic
bytes_dynamic_concat(BytesDynamic a, BytesDynamic b, BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(a.initialized);
  bytes_dynamic_init_check(b.initialized);
  BytesDynamic d = {0};
  d.offset = pool->pool_cursor;
  d.length = a.length + b.length;
  d.capacity = d.length;
  d.initialized = 1;
  memcpy(&pool->pool[d.offset], &pool->pool[a.offset], a.length);
  memcpy(&pool->pool[d.offset + a.length], &pool->pool[b.offset], b.length);
  pool->pool_cursor += d.length;
  return d;
}

BytesDynamic bytes_dynamic_copy(const BytesDynamic *src, BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  BytesDynamic d = {0};
  d.offset = pool->pool_cursor;
  d.length = src->length;
  d.capacity = src->length;
  d.initialized = 1;
  memcpy(&pool->pool[d.offset], &pool->pool[src->offset], src->length);
  pool->pool_cursor += d.length;
  return d;
}

void bytes_static_set(BytesStatic *b, size_t index, BytesStatic value)
{
__ESBMC_HIDE:;
  b->data[index] = value.data[0];
}

void bytes_dynamic_set(
  BytesDynamic *b,
  size_t index,
  BytesStatic value,
  BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  bytes_dynamic_ensure_capacity(b, index + 1, pool);
  pool->pool[b->offset + index] = value.data[0];
  if (index >= b->length)
  {
    b->length = index + 1;
  }
}

BytesStatic bytes_static_get(const BytesStatic *b, size_t index)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
  r.data[0] = b->data[index];
  r.length = 1;
  return r;
}

BytesStatic
bytes_dynamic_get(const BytesDynamic *b, const BytesPool *pool, size_t index)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  bytes_dynamic_bounds_check(index, b->length);
  BytesStatic r = {0};
  r.data[0] = pool->pool[b->offset + index];
  r.length = 1;
  return r;
}

bool bytes_static_equal(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  if (a->length != b->length)
    return false;
  return memcmp(a->data, b->data, a->length) == 0;
}

bool bytes_dynamic_equal(
  const BytesDynamic *a,
  const BytesDynamic *b,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(a->initialized);
  bytes_dynamic_init_check(b->initialized);
  if (a->length != b->length)
    return false;
  return memcmp(&pool->pool[a->offset], &pool->pool[b->offset], a->length) == 0;
}

uint256_t
bytes_dynamic_to_mapping_key(const BytesDynamic *b, const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  uint256_t result = 0;
  for (size_t i = 0; i < b->length; i++)
  {
    result = (result << 8) | pool->pool[b->offset + i];
  }
  result |= ((uint256_t)b->length) << 248;
  return result;
}

void bytes_dynamic_push(BytesDynamic *b, unsigned char value, BytesPool *pool)
{
__ESBMC_HIDE:;
  if (!b->initialized)
  {
    b->offset = pool->pool_cursor;
    b->length = 0;
    b->capacity = 4;
    b->initialized = 1;
    pool->pool_cursor += b->capacity;
  }
  bytes_dynamic_ensure_capacity(b, b->length + 1, pool);
  pool->pool[b->offset + b->length] = value;
  b->length++;
}

void bytes_dynamic_pop(BytesDynamic *b, BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  bytes_dynamic_bounds_check(0, b->length);
  b->length--;
}

uint256_t bytes_dynamic_to_uint(const BytesDynamic *b, const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  uint256_t result = 0;
  for (size_t i = 0; i < b->length; i++)
  {
    result = (result << 8) | pool->pool[b->offset + i];
  }
  return result;
}

char *bytes_static_to_string(const BytesStatic *b)
{
__ESBMC_HIDE:;
  char *out = (char *)malloc(b->length + 1);
  for (size_t i = 0; i < b->length; i++)
  {
    out[i] = (char)b->data[i];
  }
  out[b->length] = '\0';
  return out;
}

char *bytes_dynamic_to_string(const BytesDynamic *b, const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  char *out = (char *)malloc(b->length + 1);
  for (size_t i = 0; i < b->length; i++)
  {
    out[i] = (char)pool->pool[b->offset + i];
  }
  out[b->length] = '\0';
  return out;
}

BytesStatic bytes_static_extend(const BytesStatic *src, size_t new_len)
{
__ESBMC_HIDE:;
  BytesStatic out = {0};
  memcpy(out.data, src->data, src->length);
  memset(out.data + src->length, 0, new_len - src->length);
  out.length = new_len;
  return out;
}

BytesStatic bytes_static_resize(const BytesStatic *src, size_t new_len)
{
__ESBMC_HIDE:;
  if (new_len == src->length)
  {
    return *src;
  }
  else if (new_len < src->length)
  {
    return bytes_static_truncate(src, new_len);
  }
  else
  {
    return bytes_static_extend(src, new_len);
  }
}

BytesStatic bytes_static_extend_from_dynamic(
  const BytesDynamic *src,
  size_t new_len,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  BytesStatic b = {0};
  memcpy(b.data, &pool->pool[src->offset], src->length);
  memset(b.data + src->length, 0, new_len - src->length);
  b.length = new_len;
  return b;
}

BytesStatic bytes_static_resize_from_dynamic(
  const BytesDynamic *src,
  size_t new_len,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  if (new_len == src->length)
  {
    BytesStatic b = {0};
    memcpy(b.data, &pool->pool[src->offset], new_len);
    b.length = new_len;
    return b;
  }
  else if (new_len < src->length)
  {
    return bytes_static_truncate_from_dynamic(src, new_len, pool);
  }
  else
  {
    return bytes_static_extend_from_dynamic(src, new_len, pool);
  }
}

BytesPool bytes_pool_init(unsigned char *pool_data)
{
__ESBMC_HIDE:;
  BytesPool pool = {pool_data, 0};
  return pool;
}
