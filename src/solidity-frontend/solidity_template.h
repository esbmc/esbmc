/*
  The template/library for Solidity built-in variables, function and data structure
*/

#include <nlohmann/json.hpp>
#include <unordered_set>

#ifndef SOLIDITY_TEMPLATE_H_
#define SOLIDITY_TEMPLATE_H_

namespace SolidityTemplate
{

/// header & typedef

const std::string sol_header = R"(
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
)";

/*
  uint == uint256_t
  bytes == uint256_t
  bytes32 == uint256_t
  address == address_t
*/
const std::string sol_typedef = R"(
typedef unsigned _ExtInt(256) uint256_t;
typedef unsigned _ExtInt(160) address_t;
)";

/// Variables
// the value of these variables need to be set to rand afterwards

const std::string sol_msg = R"(
uint256_t msg_data;
address_t msg_address;
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
uint256_t blockhash();
)";

const std::string gasleft = R"(
uint256_t gasleft();
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
	return (x + y) % k;
}

uint256_t mulmod(uint256_t x, uint256_t y, uint256_t k)
{
	return (x * y) % k;
}

uint256_t keccak256();
uint256_t sha256();
address_t ripemd160();
address_t ecrecover();
)";

const std::string sol_string = R"(
char* string_concat(char *x, char *y)
{
	strcat(x, y);
	return x;
}
)";

const std::string sol_byte = R"(
void byte_concat();
)";

const std::string sol_funcs =
  blockhash + gasleft + sol_abi + sol_math + sol_string + sol_byte;

/// data structure

/* https://github.com/rxi/map */
const std::string sol_mapping = R"(
#ifndef MAP_H
#define MAP_H

struct map_node_t;
typedef struct map_node_t map_node_t;

typedef struct
{
	map_node_t **buckets;
	unsigned nbuckets, nnodes;
} map_base_t;

typedef struct
{
	unsigned bucketidx;
	map_node_t *node;
} map_iter_t;

#define map_t(T)         \
	struct               \
	{                    \
		map_base_t base; \
		T *ref;          \
		T tmp;           \
	}

#define map_init(m) \
	memset(m, 0, sizeof(*(m)))

#define map_deinit(m) \
	map_deinit_(&(m)->base)

#define map_get(m, key) \
	((m)->ref = map_get_(&(m)->base, key))

#define map_set(m, key, value) \
	((m)->tmp = (value),       \
	 map_set_(&(m)->base, key, &(m)->tmp, sizeof((m)->tmp)))

#define map_remove(m, key) \
	map_remove_(&(m)->base, key)

#define map_iter(m) \
	map_iter_()

#define map_next(m, iter) \
	map_next_(&(m)->base, iter)

void map_deinit_(map_base_t *m);
void *map_get_(map_base_t *m, const char *key);
int map_set_(map_base_t *m, const char *key, void *value, int vsize);
void map_remove_(map_base_t *m, const char *key);
map_iter_t map_iter_(void);
const char *map_next_(map_base_t *m, map_iter_t *iter);

typedef map_t(void *) map_void_t;
typedef map_t(char *) map_str_t;
typedef map_t(int) map_int_t;
typedef map_t(char) map_char_t;

struct map_node_t
{
	unsigned hash;
	void *value;
	map_node_t *next;
};

static unsigned map_hash(const char *str)
{
	unsigned hash = 5381;
	while (*str)
	{
		hash = ((hash << 5) + hash) ^ *str++;
	}
	return hash;
}

static map_node_t *map_newnode(const char *key, void *value, int vsize)
{
	map_node_t *node;
	int ksize = strlen(key) + 1;
	int voffset = ksize + ((sizeof(void *) - ksize) % sizeof(void *));
	node = malloc(sizeof(*node) + voffset + vsize);
	if (!node)
		return NULL;
	memcpy(node + 1, key, ksize);
	node->hash = map_hash(key);
	node->value = ((char *)(node + 1)) + voffset;
	memcpy(node->value, value, vsize);
	return node;
}

static int map_bucketidx(map_base_t *m, unsigned hash)
{
	return hash & (m->nbuckets - 1);
}

static void map_addnode(map_base_t *m, map_node_t *node)
{
	int n = map_bucketidx(m, node->hash);
	node->next = m->buckets[n];
	m->buckets[n] = node;
}

static int map_resize(map_base_t *m, int nbuckets)
{
	map_node_t *nodes, *node, *next;
	map_node_t **buckets;
	int i;
	nodes = NULL;
	i = m->nbuckets;
	while (i--)
	{
		node = (m->buckets)[i];
		while (node)
		{
			next = node->next;
			node->next = nodes;
			nodes = node;
			node = next;
		}
	}
	/* Reset buckets */
	buckets = realloc(m->buckets, sizeof(*m->buckets) * nbuckets);
	if (buckets != NULL)
	{
		m->buckets = buckets;
		m->nbuckets = nbuckets;
	}
	if (m->buckets)
	{
		memset(m->buckets, 0, sizeof(*m->buckets) * m->nbuckets);
		/* Re-add nodes to buckets */
		node = nodes;
		while (node)
		{
			next = node->next;
			map_addnode(m, node);
			node = next;
		}
	}
	/* Return error code if realloc() failed */
	return (buckets == NULL) ? -1 : 0;
}

static map_node_t **map_getref(map_base_t *m, const char *key)
{
	unsigned hash = map_hash(key);
	map_node_t **next;
	if (m->nbuckets > 0)
	{
		next = &m->buckets[map_bucketidx(m, hash)];
		while (*next)
		{
			if ((*next)->hash == hash && !strcmp((char *)(*next + 1), key))
			{
				return next;
			}
			next = &(*next)->next;
		}
	}
	return NULL;
}

void map_deinit_(map_base_t *m)
{
	map_node_t *next, *node;
	int i;
	i = m->nbuckets;
	while (i--)
	{
		node = m->buckets[i];
		while (node)
		{
			next = node->next;
			free(node);
			node = next;
		}
	}
	free(m->buckets);
}

void *map_get_(map_base_t *m, const char *key)
{
	map_node_t **next = map_getref(m, key);
	return next ? (*next)->value : NULL;
}

int map_set_(map_base_t *m, const char *key, void *value, int vsize)
{
	int n, err;
	map_node_t **next, *node;
	next = map_getref(m, key);
	if (next)
	{
		memcpy((*next)->value, value, vsize);
		return 0;
	}
	node = map_newnode(key, value, vsize);
	if (node == NULL)
		goto fail;
	if (m->nnodes >= m->nbuckets)
	{
		n = (m->nbuckets > 0) ? (m->nbuckets << 1) : 1;
		err = map_resize(m, n);
		if (err)
			goto fail;
	}
	map_addnode(m, node);
	m->nnodes++;
	return 0;
fail:
	if (node)
		free(node);
	return -1;
}

void map_remove_(map_base_t *m, const char *key)
{
	map_node_t *node;
	map_node_t **next = map_getref(m, key);
	if (next)
	{
		node = *next;
		*next = (*next)->next;
		free(node);
		m->nnodes--;
	}
}

map_iter_t map_iter_(void)
{
	map_iter_t iter;
	iter.bucketidx = -1;
	iter.node = NULL;
	return iter;
}

const char *map_next_(map_base_t *m, map_iter_t *iter)
{
	if (iter->node)
	{
		iter->node = iter->node->next;
		if (iter->node == NULL)
			goto nextBucket;
	}
	else
	{
	nextBucket:
		do
		{
			if (++iter->bucketidx >= m->nbuckets)
			{
				return NULL;
			}
			iter->node = m->buckets[iter->bucketidx];
		} while (iter->node == NULL);
	}
	return (char *)(iter->node + 1);
}
#endif
)";

// combination
const std::string sol_library =
  sol_header + sol_typedef + sol_vars + sol_funcs + sol_mapping;

}; // namespace SolidityTemplate

#endif
