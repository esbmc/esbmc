/*
  The template/library for Solidity built-in variables, function and data structure
*/

#include <nlohmann/json.hpp>
#include <unordered_set>

#ifndef SOLIDITY_TEMPLATE_H_
#  define SOLIDITY_TEMPLATE_H_

namespace SolidityTemplate
{
/// header & typedef
const std::string sol_header = R"(
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
// #include <string>
// #include <math.h>
)";

/*
  uint == uint256_t
  bytes == uint256_t
  bytes32 == uint256_t
  address == address_t
*/
const std::string sol_typedef = R"(
#if defined(__clang__)  // Ensure we are using Clang
    #if __clang_major__ >= 16
        #define BIGINT(bits) _BitInt(bits)
    #elif __clang_major__ >= 11 && __clang_major__ <= 13
        #define BIGINT(bits) _ExtInt(bits)
    #else
        #error "Unsupported Clang version: ExtInt and BigInt are not properly supported. Please use Clang 11-13 for ExtInt or Clang 16+ for BigInt."
    #endif
#else
    #error "This code requires Clang to compile."
#endif

typedef BIGINT(256) int256_t;
typedef unsigned BIGINT(256) uint256_t;
typedef unsigned BIGINT(160) address_t;

struct sol_llc_ret
{
  bool x;
  unsigned int y;
};
)";

/// Variables
// the value of these variables need to be set to rand afterwards
const std::string sol_msg = R"(
uint256_t msg_data;
address_t msg_sender;
__uint32_t msg_sig;
uint256_t msg_value;
)";

const std::string sol_tx = R"(
uint256_t tx_gasprice;
address_t tx_origin;
)";

const std::string sol_block = R"(
uint256_t block_basefee;
uint256_t block_chainid;
address_t block_coinbase;
uint256_t block_difficulty;
uint256_t block_gaslimit;
uint256_t block_number;
uint256_t block_prevrandao;
uint256_t block_timestamp;
)";

const std::string sol_vars = sol_msg + sol_tx + sol_block;

/// functions
// if the function does not currently have an actual implement,
// leave the params empty.
const std::string blockhash = R"(
uint256_t blockhash(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}
)";

const std::string gasleft = R"(
unsigned int _gaslimit;
void gasConsume()
{
__ESBMC_HIDE:;
  unsigned int consumed = nondet_uint();
  __ESBMC_assume(consumed > 0 && consumed <= _gaslimit);
  _gaslimit -= consumed;
}
uint256_t gasleft()
{
__ESBMC_HIDE:;
  gasConsume(); // always less
  return (uint256_t)_gaslimit;
}
)";

const std::string sol_abi = R"(
uint256_t abi_encode();
uint256_t abi_encodePacked();
uint256_t abi_encodeWithSelector();
uint256_t abi_encodeWithSignature();
uint256_t abi_encodeCall();
)";

const std::string sol_math = R"(
uint256_t addmod(uint256_t x, uint256_t y, uint256_t k)
{
__ESBMC_HIDE:;
	return (x + y) % k;
}

uint256_t mulmod(uint256_t x, uint256_t y, uint256_t k)
{
__ESBMC_HIDE:;
	return (x * y) % k;
}

uint256_t keccak256(uint256_t x)
{
__ESBMC_HIDE:;
  return  x;
}

uint256_t sha256(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}
address_t ripemd160(uint256_t x)
{
__ESBMC_HIDE:;
  // UNSAT abstraction
  return (address_t)x;
}
address_t ecrecover(uint256_t hash, unsigned int v, uint256_t r, uint256_t s)
{
__ESBMC_HIDE:;
  return (address_t)hash;
}

// uint256_t _pow(unsigned int base, unsigned int exp) {
// __ESBMC_HIDE:;
//   uint256_t result = 1;
//   uint256_t b = base;

//   while (exp > 0) {
//     if (exp & 1)
//       result *= b;
//     b *= b;
//     exp >>= 1;
//   }

//   return result;
// }
double pow(double x, double y);
)";

const std::string sol_string = R"(
char* string_concat(char *x, char *y)
{
__ESBMC_HIDE:;
	strncat(x, y, 256);
	return x;
}
)";

const std::string sol_byte = R"(
typedef struct BytesPool {
    unsigned char* pool;
    size_t pool_cursor;
} BytesPool;

typedef struct BytesStatic {
    unsigned char data[32];
    size_t length;
} BytesStatic;

typedef struct BytesDynamic {
    size_t offset;
    size_t length;
    int initialized;
} BytesDynamic;

unsigned char hex_char_to_nibble(char c) {
__ESBMC_HIDE:;
    if ('0' <= c && c <= '9') return c - '0';
    else if ('a' <= tolower(c) && tolower(c) <= 'f') return tolower(c) - 'a' + 10;
    else
      abort();
    return 0;
}

BytesStatic bytes_static_from_hex(const char* hex_str) {
__ESBMC_HIDE:;
    BytesStatic b = {0};
    size_t hex_len = strlen(hex_str) - 2;
    b.length = hex_len / 2;
    for (size_t i = 0; i < b.length; i++) {
        unsigned char high = hex_char_to_nibble(hex_str[2 + i * 2]);
        unsigned char low = hex_char_to_nibble(hex_str[2 + i * 2 + 1]);
        b.data[i] = (high << 4) | low;
    }
    return b;
}

BytesStatic bytes_static_from_string(const char* str) {
__ESBMC_HIDE:;
    size_t len = strlen(str);
    BytesStatic b = {0};
    for (size_t i = 0; i < len; i++) {
        b.data[i] = str[i];
    }
    b.length = len;
    return b;
}

BytesStatic bytes_static_truncate(const BytesStatic* src, size_t new_len) {
__ESBMC_HIDE:;
    BytesStatic b = {0};
    for (size_t i = 0; i < new_len; i++) {
        b.data[i] = src->data[i];
    }
    b.length = new_len;
    return b;
}

BytesStatic bytes_static_and(const BytesStatic* a, const BytesStatic* b) {
__ESBMC_HIDE:;
    BytesStatic r = {0};
    for (size_t i = 0; i < a->length; i++) {
        r.data[i] = a->data[i] & b->data[i];
    }
    r.length = a->length;
    return r;
}

BytesStatic bytes_static_or(const BytesStatic* a, const BytesStatic* b) {
__ESBMC_HIDE:;
    BytesStatic r = {0};
    for (size_t i = 0; i < a->length; i++) {
        r.data[i] = a->data[i] | b->data[i];
    }
    r.length = a->length;
    return r;
}

BytesStatic bytes_static_xor(const BytesStatic* a, const BytesStatic* b) {
__ESBMC_HIDE:;
    BytesStatic r = {0};
    for (size_t i = 0; i < a->length; i++) {
        r.data[i] = a->data[i] ^ b->data[i];
    }
    r.length = a->length;
    return r;
}

uint256_t bytes_static_to_uint(const BytesStatic* b) {
__ESBMC_HIDE:;
    uint256_t result = 0;
    for (size_t i = 0; i < b->length; i++) {
        result = (result << 8) | b->data[i];
    }
    return result;
}

BytesStatic bytes_static_from_uint(uint256_t val, size_t len) {
__ESBMC_HIDE:;
    BytesStatic b = {0};
    for (size_t i = 0; i < len; i++) {
        b.data[len - 1 - i] = val & 0xFF;
        val >>= 8;
    }
    b.length = len;
    return b;
}

BytesStatic bytes_static_shl(const BytesStatic* src, unsigned shift_bits) {
__ESBMC_HIDE:;
    uint256_t val = bytes_static_to_uint(src);
    val <<= shift_bits;
    return bytes_static_from_uint(val, src->length);
}

BytesStatic bytes_static_shr(const BytesStatic* src, unsigned shift_bits) {
__ESBMC_HIDE:;
    uint256_t val = bytes_static_to_uint(src);
    val >>= shift_bits;
    return bytes_static_from_uint(val, src->length);
}

uint256_t bytes_static_to_mapping_key(const BytesStatic* b) {
__ESBMC_HIDE:;
    return ((uint256_t)b->length << 248) | bytes_static_to_uint(b);
}

BytesStatic bytes_static_init_zero(size_t len) {
__ESBMC_HIDE:;
    BytesStatic b = {0};
    b.length = len;
    for (size_t i = 0; i < len; i++) {
        b.data[i] = 0;
    }
    return b;
}

BytesDynamic bytes_dynamic_init_zero(size_t len, BytesPool* pool) {
__ESBMC_HIDE:;
    BytesDynamic b = {0};
    b.offset = pool->pool_cursor;
    b.length = len;
    b.initialized = 1;
    for (size_t i = 0; i < len; i++) {
      pool->pool[b.offset + i] = 0;
    }
    pool->pool_cursor += len;
    return b;
}

void bytes_dynamic_init(BytesDynamic* b, const unsigned char* input, size_t len, BytesPool* pool) {
__ESBMC_HIDE:;
    b->offset = pool->pool_cursor;
    b->length = len;
    b->initialized = 1;
    for (size_t i = 0; i < len; i++) {
        pool->pool[b->offset + i] = input[i];
    }
    pool->pool_cursor += len;
}

BytesDynamic bytes_dynamic_from_static(const BytesStatic* s, BytesPool* pool) {
__ESBMC_HIDE:;
    BytesDynamic b = {0};
    bytes_dynamic_init(&b, s->data, s->length, pool);
    return b;
}

BytesDynamic bytes_dynamic_from_string(const char* str, BytesPool* pool) {
__ESBMC_HIDE:;
    BytesDynamic b = {0};
    bytes_dynamic_init(&b, (const unsigned char*)str, strlen(str), pool);
    return b;
}

BytesDynamic bytes_dynamic_from_hex(const char* hex_str, BytesPool* pool) {
__ESBMC_HIDE:;
    size_t hex_len = strlen(hex_str) - 2;
    size_t byte_len = hex_len / 2;

    unsigned char tmp[32] = {0};
    for (size_t i = 0; i < byte_len; i++) {
        unsigned char high = hex_char_to_nibble(hex_str[2 + i * 2]);
        unsigned char low = hex_char_to_nibble(hex_str[2 + i * 2 + 1]);
        tmp[i] = (high << 4) | low;
    }

    BytesDynamic b = {0};
    bytes_dynamic_init(&b, tmp, byte_len, pool);
    return b;
}

BytesStatic bytes_static_truncate_from_dynamic(const BytesDynamic* src, size_t new_len, const BytesPool* pool) {
__ESBMC_HIDE:;
    BytesStatic b = {0};
    for (size_t i = 0; i < new_len; i++) {
        b.data[i] = pool->pool[src->offset + i];
    }
    b.length = new_len;
    return b;
}

BytesDynamic bytes_dynamic_concat(const BytesDynamic* a, const BytesDynamic* b, BytesPool* pool) {
__ESBMC_HIDE:;
    BytesDynamic d = {0};
    d.offset = pool->pool_cursor;
    d.length = a->length + b->length;
    d.initialized = 1;
    for (size_t i = 0; i < a->length; i++) {
        pool->pool[d.offset + i] = pool->pool[a->offset + i];
    }
    for (size_t i = 0; i < b->length; i++) {
        pool->pool[d.offset + a->length + i] = pool->pool[b->offset + i];
    }
    pool->pool_cursor += d.length;
    return d;
}

BytesDynamic bytes_dynamic_copy(const BytesDynamic* src, BytesPool* pool) {
__ESBMC_HIDE:;
    BytesDynamic d = {0};
    d.offset = pool->pool_cursor;
    d.length = src->length;
    d.initialized = 1;
    for (size_t i = 0; i < src->length; i++) {
        pool->pool[d.offset + i] = pool->pool[src->offset + i];
    }
    pool->pool_cursor += d.length;
    return d;
}

void bytes_static_set(BytesStatic* b, size_t index, BytesStatic value) {
__ESBMC_HIDE:;
    b->data[index] = value.data[0];
}

void bytes_dynamic_set(BytesDynamic* b, size_t index, BytesStatic value, BytesPool* pool) {
__ESBMC_HIDE:;
    pool->pool[b->offset + index] = value.data[0];
}

BytesStatic bytes_static_get(const BytesStatic* b, size_t index) {
__ESBMC_HIDE:;
    BytesStatic r = {0};
    r.data[0] = b->data[index];
    r.length = 1;
    return r;
}

BytesStatic bytes_dynamic_get(const BytesDynamic* b, const BytesPool* pool, size_t index) {
__ESBMC_HIDE:;
    BytesStatic r = {0};
    r.data[0] = pool->pool[b->offset + index];
    r.length = 1;
    return r;
}

bool bytes_static_equal(const BytesStatic* a, const BytesStatic* b) {
__ESBMC_HIDE:;
    if (a->length != b->length) return false;
    for (size_t i = 0; i < a->length; i++) {
        if (a->data[i] != b->data[i]) return false;
    }
    return true;
}

bool bytes_dynamic_equal(const BytesDynamic* a, const BytesDynamic* b, const BytesPool* pool) {
__ESBMC_HIDE:;
    if (a->length != b->length) return false;
    for (size_t i = 0; i < a->length; i++) {
        if (pool->pool[a->offset + i] != pool->pool[b->offset + i]) return false;
    }
    return true;
}

uint256_t bytes_dynamic_to_mapping_key(const BytesDynamic* b, const BytesPool* pool) {
__ESBMC_HIDE:;
    uint256_t result = 0;
    for (size_t i = 0; i < b->length; i++) {
        result = (result << 8) | pool->pool[b->offset + i];
    }
    result |= ((uint256_t)b->length) << 248;
    return result;
}

void bytes_dynamic_push(BytesDynamic* b, unsigned char value, BytesPool* pool) {
__ESBMC_HIDE:;
    pool->pool[b->offset + b->length] = value;
    b->length++;
}

void bytes_dynamic_pop(BytesDynamic* b, BytesPool* pool) {
__ESBMC_HIDE:;
    b->length--;
}

BytesPool bytes_pool_init(unsigned char* pool_data) {
__ESBMC_HIDE:;
    BytesPool pool = { pool_data, 0 };
    return pool;
}
)";

const std::string sol_destruct = R"(
void selfdestruct()
{
__ESBMC_HIDE:;
  exit(0);
}
)";

const std::string sol_funcs = blockhash + gasleft + sol_abi + sol_math +
                              sol_string + sol_byte + sol_destruct;

/// data structure

/* https://github.com/rxi/map */
const std::string sol_mapping = R"(
struct _ESBMC_Mapping
{
  address_t addr : 160;
  uint256_t key : 256;
  void *value;
  struct _ESBMC_Mapping *next;
};

struct mapping_t
{
  struct _ESBMC_Mapping *base;
  address_t addr : 160;
};

void *map_get_raw(struct _ESBMC_Mapping a[], address_t addr, uint256_t key)
{
__ESBMC_HIDE:;
  struct _ESBMC_Mapping *cur = a[key].next;
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
  n->next = a[key].next;
  a[key].next = n;
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
)";

// used when there is no NewExpression in the source json.
const std::string sol_mapping_fast = R"(
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
  struct _ESBMC_Mapping_fast *cur = a[key].next;
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
  n->next = a[key].next;
  a[key].next = n;
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
)";

const std::string sol_array = R"(
// Node structure for linked list
typedef struct ArrayNode {
    void *array_ptr;        // Pointer to the stored array
    size_t length;          // Length of the array
    struct ArrayNode *next; // Pointer to the next node
} ArrayNode;

// Head of the linked list
ArrayNode *array_list_head = NULL;

/**
 * Stores/updates an array and its length.
 * If the array already exists, it updates the length.
 */
void _ESBMC_store_array(void *array, size_t length) {
__ESBMC_HIDE:;
    // Check if array already exists in the list
    ArrayNode *current = array_list_head;
    while (current != NULL) {
        if (current->array_ptr == array) { // Found existing array
            current->length = length; // Update length
            return;
        }
        current = current->next;
    }

    // Create a new node
    ArrayNode *new_node = (ArrayNode *)malloc(sizeof(ArrayNode));
    new_node->array_ptr = array;
    new_node->length = length;
    new_node->next = array_list_head; // Insert at head
    array_list_head = new_node;
}

/**
 * Fetches the length of a stored array.
 * Returns 0 if the array is not found.
 */
unsigned int _ESBMC_array_length(void *array) {
__ESBMC_HIDE:;
    ArrayNode *current = array_list_head;
    while (current != NULL) {
        if (current->array_ptr == array) {
            return current->length;
        }
        current = current->next;
    }
    return 0;
}


void *_ESBMC_arrcpy(void *from_array, size_t from_size, size_t size_of)
{
__ESBMC_HIDE:;
  // assert(from_size != 0);
  if(from_array == NULL || size_of == 0 || from_size == 0)
    abort();

  void *to_array = (void *)calloc(from_size, size_of);
  memcpy(to_array, from_array, from_size * size_of);
  return to_array;
}

void _ESBMC_array_push(void *array, void *element, size_t size_of_element) {
__ESBMC_HIDE:;
    if (array == NULL || element == NULL || size_of_element == 0)
        abort();

    ArrayNode *current = array_list_head;
    while (current != NULL) {
        if (current->array_ptr == array) {
            size_t new_length = current->length + 1;
            void *new_array = realloc(array, new_length * size_of_element);
            if (new_array == NULL)
                abort();
            
            memcpy((char *)new_array + (current->length * size_of_element), element, size_of_element);
            current->array_ptr = new_array;
            current->length = new_length;
            return;
        }
        current = current->next;
    }

    // Array not found, treat as new
    void *new_array = malloc(size_of_element);
    if (new_array == NULL)
        abort();
    memcpy(new_array, element, size_of_element);
    _ESBMC_store_array(new_array, 1);
}

void _ESBMC_array_pop(void *array, size_t size_of_element) {
__ESBMC_HIDE:;
    if (array == NULL || size_of_element == 0)
        abort();

    ArrayNode *current = array_list_head;
    while (current != NULL) {
        if (current->array_ptr == array) {
            if (current->length == 0)
                abort(); // Cannot pop from empty array

            current->length -= 1;

            if (current->length == 0) {
                free(current->array_ptr);
                current->array_ptr = NULL;
            } else {
                void *new_array = realloc(current->array_ptr, current->length * size_of_element);
                if (new_array != NULL)
                    current->array_ptr = new_array;
            }
            return;
        }
        current = current->next;
    }
}
)";

const std::string sol_unit = R"(
// ether
uint256_t _ESBMC_wei(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}
uint256_t _ESBMC_gwei(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000; // 10^9
}
uint256_t _ESBMC_szabo(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000000; // 10^12
}
uint256_t _ESBMC_finney(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000000000; // 10^15
}
uint256_t _ESBMC_ether(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000000000000; // 10^18
}

// time
uint256_t _ESBMC_seconds(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}
uint256_t _ESBMC_minutes(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)60;
}
uint256_t _ESBMC_hours(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)3600; // 60 * 60
}
uint256_t _ESBMC_days(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)86400; // 24 * 3600
}
uint256_t _ESBMC_weeks(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)604800; // 7 * 86400
}
uint256_t _ESBMC_years(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)31536000; // 365 * 86400
} 
)";

/// external library
// itoa

const std::string sol_itoa = R"(
char get_char(int digit)
{
__ESBMC_HIDE:;
    char charstr[] = "0123456789ABCDEF";
    return charstr[digit];
}
void sol_rev(char *p)
{
__ESBMC_HIDE:;
	char *q = &p[strlen(p) - 1];
	char *r = p;
	for (; q > r; q--, r++)
	{
		char s = *q;
		*q = *r;
		*r = s;
	}
}
char *i256toa(int256_t value)
{
__ESBMC_HIDE:;
	// we might have memory leak as we will not free this afterwards
	char *str = (char *)malloc(256 * sizeof(char));
	int256_t base = (int256_t)10;
	unsigned short count = 0;
	bool flag = true;

	if (value < (int256_t)0 && base == (int256_t)10)
	{
		flag = false;
	}
	if (value == (int256_t)0)
	{
		str[count] = '\0';
		return str;
	}
	while (value != (int256_t)0)
	{
		int256_t dig = value % base;
		value -= dig;
		value /= base;

		if (flag == true)
			str[count] = get_char(dig);
		else
			str[count] = get_char(-dig);
		count++;
	}
	if (flag == false)
	{
		str[count] = '-';
		count++;
	}
	str[count] = 0;
	sol_rev(str);
	return str;
}

char *u256toa(uint256_t value)
{
__ESBMC_HIDE:;
	char *str = (char *)malloc(256 * sizeof(char));
	uint256_t base = (uint256_t)10;
	unsigned short count = 0;
	if (value == (uint256_t)0)
	{
		str[count] = '\0';
		return str;
	}
	while (value != (uint256_t)0)
	{
		uint256_t dig = value % base;
		value -= dig;
		value /= base;
		str[count] = get_char(dig);
		count++;
	}
	str[count] = 0;
	sol_rev(str);
	return str;
}
)";

// string2hex
const std::string sol_str2hex = R"(
char *decToHexa(int n)
{
__ESBMC_HIDE:;
    char *hexaDeciNum = (char *)malloc(256 * sizeof(char));
    hexaDeciNum[0] = '\0';
    int i = 0;
    while (n != 0)
    {
        int temp = 0;
        temp = n % 16;
        if (temp < 10)
        {
            hexaDeciNum[i] = temp + 48;
            i++;
        }
        else
        {
            hexaDeciNum[i] = temp + 55;
            i++;
        }

        n /= 16;
    }
    char *ans = (char *)malloc(256 * sizeof(char));
    ans[0] = '\0';
    int pos = 0;
    for (int j = i - 1; j >= 0; j--)
    {
        ans[pos] = (char)hexaDeciNum[j];
        pos++;
    }
    ans[pos] = '\0';
    return ans;
}
char *ASCIItoHEX(const char *ascii)
{
__ESBMC_HIDE:;
    char *hex = (char *)malloc(256 * sizeof(char));
    hex[0] = '\0';
    for (int i = 0; i < strlen(ascii); i++)
    {
        char ch = ascii[i];
        int tmp = (int)ch;
        char *part = decToHexa(tmp);
        strcat(hex, part);
    }
    return hex;
}
uint256_t hexdec(const char *hex)
{
__ESBMC_HIDE:;
    /*https://stackoverflow.com/questions/10324/convert-a-hexadecimal-string-to-an-integer-efficiently-in-c*/

    static const long hextable[] = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    uint256_t ret = 0;
    while (*hex && ret >= (uint256_t)0)
    {
        ret = (ret << (uint256_t)4) | (uint256_t)hextable[*hex++];
    }
    return ret;
}
uint256_t str2uint(const char *str)
{
__ESBMC_HIDE:;
    return hexdec(ASCIItoHEX(str));
}

// string assign
void _str_assign(char **str1, const char *str2) {
__ESBMC_HIDE:;
    // Ensure str1 is a valid pointer (not NULL)
    if (str1 == NULL) {
        return;  // Early exit if str1 is invalid
    }
    // Free *str1 only if it was previously allocated (non-NULL)
    // if (*str1 != NULL) {
    //     free(*str1);
    // }
    
    // If str2 is NULL, set *str1 to NULL (avoid dangling pointers)
    if (str2 == NULL) {
        *str1 = NULL;
        return;
    }
    size_t len = strlen(str2);
    if (len == SIZE_MAX) {  // Would overflow when adding 1
        *str1 = NULL;  // Or handle error differently
        return;
    }
    *str1 = (char *)malloc(len + 1);  
    strcpy(*str1, str2);  // force malloc success
}
__attribute__((annotate("__ESBMC_inf_size"))) char _ESBMC_rand_str[1];
char *nondet_string() {
__ESBMC_HIDE:;
    size_t len = nondet_uint();
    // __ESBMC_assume(len < SIZE_MAX);  

    for (size_t i = 0; i < len; ++i) {
        _ESBMC_rand_str[i] = nondet_char();
        __ESBMC_assume(_ESBMC_rand_str[i] != '\0');
    }
    _ESBMC_rand_str[len] = '\0'; 
    return _ESBMC_rand_str;
}
)";

// get unique random address
const std::string sol_uqAddr = R"(
__attribute__((annotate("__ESBMC_inf_size"))) address_t sol_addr_array[1];
__attribute__((annotate("__ESBMC_inf_size"))) void *sol_obj_array[1];
__attribute__((annotate("__ESBMC_inf_size"))) const char *sol_cname_array[1];
unsigned int sol_max_cnt;

int _ESBMC_get_addr_array_idx(address_t tgt)
{
__ESBMC_HIDE:;
    if(tgt == (address_t)0)
      return -1;

    for (unsigned int i = 0; i < sol_max_cnt; i++)
    {
        if ((address_t)sol_addr_array[i] == (address_t)tgt)
            return i;
    }
    return -1;
}
bool _ESBMC_cmp_cname(const char *c_1, const char *c_2)
{
__ESBMC_HIDE:;
    return strcmp(c_1, c_2) == 0;
}
void *_ESBMC_get_obj(address_t addr, const char *cname)
{
__ESBMC_HIDE:;
    int idx = _ESBMC_get_addr_array_idx(addr);
    if (idx == -1)
        // this means it's not previously stored
        return NULL;
    if (_ESBMC_cmp_cname(sol_cname_array[idx], cname))
        return sol_obj_array[idx];
    return NULL;
}
void update_addr_obj(address_t addr, void *obj, const char *cname)
{
__ESBMC_HIDE:;
    // __ESBMC_assume(obj != NULL);
    sol_addr_array[sol_max_cnt] = addr;
    sol_obj_array[sol_max_cnt] = obj;
    sol_cname_array[sol_max_cnt] = cname;
    ++sol_max_cnt;
}
address_t _ESBMC_get_unique_address(void *obj, const char *cname)
{
__ESBMC_HIDE:;
    // __ESBMC_assume(obj != NULL);
    address_t tmp;
    do {
        tmp = (address_t)nondet_uint();
        if (tmp == (address_t)0)
            continue;
        if (sol_max_cnt == 0)
            break;
    } while (_ESBMC_get_addr_array_idx(tmp) == -1);
    
    update_addr_obj(tmp, obj, cname);
    return tmp;
}
const char *_ESBMC_get_nondet_cont_name(const char *c_array[], unsigned int len)
{
__ESBMC_HIDE:;
    unsigned int rand = nondet_uint() % len;
    return c_array[rand];
}
)";

// max/min value
const std::string sol_max_min = R"(
uint256_t _max(unsigned int bitwidth, bool is_signed) {
__ESBMC_HIDE:;
  __ESBMC_assume(bitwidth > 0 && bitwidth <= 256);
  if (is_signed) {
      return ((uint256_t)1 << (bitwidth - 1)) - (uint256_t)1;
  } else {
      if (bitwidth == 256) {
          return (uint256_t)-1; 
      }
      return ((uint256_t)1 << bitwidth) - (uint256_t)1;
  }
}
int256_t _min(unsigned int bitwidth, bool is_signed) {
__ESBMC_HIDE:;
  if (is_signed) {
      __ESBMC_assume(bitwidth > 0 && bitwidth <= 256);
      return -((int256_t)1 << (bitwidth - 1)); // -2^(N-1)
  } else {
      return (int256_t)0; // Min of unsigned is always 0
  }
}

unsigned int _creationCode()
{
__ESBMC_HIDE:;
  return nondet_uint();
}

unsigned int _runtimeCode()
{
__ESBMC_HIDE:;
  return nondet_uint();
}
)";

const std::string sol_mutex = R"(
void _ESBMC_check_reentrancy(const bool _ESBMC_mutex)
{
__ESBMC_HIDE:;
  if(_ESBMC_mutex)
    assert(!"Reentrancy behavior detected");
}
)";

const std::string sol_ext_library =
  sol_itoa + sol_str2hex + sol_uqAddr + sol_max_min + sol_mutex;

const std::string sol_initialize = R"(
void initialize()
{
__ESBMC_HIDE:;
// we assume it starts from an EOA
msg_data = (uint256_t)nondet_uint();
msg_sender = (address_t)nondet_uint();
msg_sig = nondet_uint();
msg_value = (uint256_t)nondet_uint();

tx_gasprice = (uint256_t)nondet_uint();
// this can only be an EOA's address
tx_origin = (address_t)nondet_uint();

block_basefee = (uint256_t)nondet_uint();
block_chainid = (uint256_t)nondet_uint();
block_coinbase = (address_t)nondet_uint();
block_difficulty = (uint256_t)nondet_uint();
block_gaslimit = (uint256_t)nondet_uint();
block_number = (uint256_t)nondet_uint();
block_prevrandao = (uint256_t)nondet_uint();
block_timestamp = (uint256_t)nondet_uint();

_gaslimit = nondet_uint();

sol_max_cnt = 0;
}
)";

const std::string sol_c_library = "extern \"C\" {" + sol_typedef + sol_vars +
                                  sol_funcs + sol_mapping + sol_mapping_fast +
                                  sol_array + sol_unit + sol_ext_library +
                                  sol_initialize + "}";

// C++
const std::string sol_cpp_string = R"(
const std::string empty_str = "";
void _streq(std::string &str1, std::string str2)
{
__ESBMC_HIDE:;
  // __ESBMC_assume(!str2.empty());
  str1 = str2;
}
std::string _tostr(const char* ptr)
{
__ESBMC_HIDE:;
  return std::string(ptr);
}
const char* _tochar(const std::string& str)
{
__ESBMC_HIDE:;
  return str.c_str();
}
)";

const std::string sol_cpp_library = sol_cpp_string;

// combination
const std::string sol_library =
  sol_header + sol_c_library; // + sol_cpp_library;

}; // namespace SolidityTemplate

#endif
