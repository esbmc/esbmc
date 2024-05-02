#include <nlohmann/json.hpp>
#include <unordered_set>

#ifndef SOLIDITY_TEMPLATE_H_
#define SOLIDITY_TEMPLATE_H_

namespace SolidityTemplate
{
inline const std::string parentheis = "()";

/// special variables
// msg
inline std::string msg_bs = "msg";
inline std::map<std::string, std::string> msg_mem = {
  {"data", "bytes"},
  {"sender", "address"},
  {"sig", "bytes4"},
  {"value", "uint"}};

// tx
inline std::string tx_bs = "tx";
inline std::map<std::string, std::string> tx_mem = {
  {"gasprice", "uint"},
  {"origin", "address"}};

// block
inline std::string block_bs = "block";
inline std::map<std::string, std::string> block_mem = {
  {"basefee", "uint"},
  {"chainid", "uint"},
  {"coinbase", "address"},
  {"difficulty", "uint"},
  {"gaslimit", "uint"},
  {"number", "uint"},
  {"prevrandao", "uint"},
  {"timestamp", "uint"}};

/// sepcial functions

inline std::map<std::string, std::string> block_hash = {
  {"blockhash", "bytes32"}};
inline std::map<std::string, std::string> gasleft = {{"gasleft", "uint"}};

// abi
inline std::string abi_bs = "abi";
inline std::map<std::string, std::string> ai_mem = {
  {"encode", "bytes"},
  {"encodePacked", "bytes"},
  {"encodeWithSelector", "bytes"},
  {"encodeWithSignature", "bytes"},
  {"encodeCall", "bytes"}}; // {"decode","tuple"},

// byte
inline std::string byte_bs = "byte";
inline std::map<std::string, std::string> byte_mem = {
  {"concat", "bytes"},
};

// string
inline std::string string_bs = "string";
inline std::map<std::string, std::string> string_mem = {{"concat", "string"}};

// addmod

inline std::map<std::string, std::string> addmod = {{"addmod", "uint"}};
inline std::map<std::string, std::string> mulmod = {{"mulmod", "uint"}};

// Library wriiten in solidity

// The idea is write the function in solidity and utilize the AST-json.
// The main benifit is that we can use the uint256/int256, which are
// not currently supported in C.

/// function body
// 1. set "stateVariable": true
// 2. add "isTemplate": true and "templateFunctionID"
// 3. make sure the Visibility is intnernal or private
/* 
    addmod(uint x, uint y, uint k) returns (uint)
    function addmod(uint x, uint y, uint k) public returns (uint)
    {
        assert(k != 0);
        return (x + y) % k;
    }
*/
inline const std::string addmod_body =
  R"({
    "isTemplate": true,
    "templateFunctionID": "sol:@F@addmod#",
    "body": {
    "id": 25,
    "nodeType": "Block",
    "src": "107:59:0",
    "statements": [
        {
            "expression": {
                "commonType": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                },
                "id": 23,
                "isConstant": false,
                "isLValue": false,
                "isPure": false,
                "lValueRequested": false,
                "leftExpression": {
                    "components": [
                        {
                            "commonType": {
                                "typeIdentifier": "t_uint256",
                                "typeString": "uint256"
                            },
                            "id": 20,
                            "isConstant": false,
                            "isLValue": false,
                            "isPure": false,
                            "lValueRequested": false,
                            "leftExpression": {
                                "id": 18,
                                "name": "x",
                                "nodeType": "Identifier",
                                "overloadedDeclarations": [],
                                "referencedDeclaration": 3,
                                "src": "149:1:0",
                                "typeDescriptions": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                }
                            },
                            "nodeType": "BinaryOperation",
                            "operator": "+",
                            "rightExpression": {
                                "id": 19,
                                "name": "y",
                                "nodeType": "Identifier",
                                "overloadedDeclarations": [],
                                "referencedDeclaration": 5,
                                "src": "153:1:0",
                                "typeDescriptions": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                }
                            },
                            "src": "149:5:0",
                            "typeDescriptions": {
                                "typeIdentifier": "t_uint256",
                                "typeString": "uint256"
                            }
                        }
                    ],
                    "id": 21,
                    "isConstant": false,
                    "isInlineArray": false,
                    "isLValue": false,
                    "isPure": false,
                    "lValueRequested": false,
                    "nodeType": "TupleExpression",
                    "src": "148:7:0",
                    "typeDescriptions": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    }
                },
                "nodeType": "BinaryOperation",
                "operator": "%",
                "rightExpression": {
                    "id": 22,
                    "name": "k",
                    "nodeType": "Identifier",
                    "overloadedDeclarations": [],
                    "referencedDeclaration": 7,
                    "src": "158:1:0",
                    "typeDescriptions": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    }
                },
                "src": "148:11:0",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                }
            },
            "functionReturnParameters": 11,
            "id": 24,
            "nodeType": "Return",
            "src": "141:18:0"
        }
    ]
},
"functionSelector": "9796df37",
"id": 26,
"implemented": true,
"kind": "function",
"modifiers": [],
"name": "addmod",
"nameLocation": "54:6:0",
"nodeType": "FunctionDefinition",
"parameters": {
    "id": 8,
    "nodeType": "ParameterList",
    "parameters": [
        {
            "constant": false,
            "id": 3,
            "mutability": "mutable",
            "name": "x",
            "nameLocation": "66:1:0",
            "nodeType": "VariableDeclaration",
            "src": "61:6:0",
            "stateVariable": true,
            "storageLocation": "default",
            "typeDescriptions": {
                "typeIdentifier": "t_uint256",
                "typeString": "uint256"
            },
            "typeName": {
                "id": 2,
                "name": "uint",
                "nodeType": "ElementaryTypeName",
                "src": "61:4:0",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                }
            },
            "visibility": "internal"
        },
        {
            "constant": false,
            "id": 5,
            "mutability": "mutable",
            "name": "y",
            "nameLocation": "74:1:0",
            "nodeType": "VariableDeclaration",
            "src": "69:6:0",
            "stateVariable": true,
            "storageLocation": "default",
            "typeDescriptions": {
                "typeIdentifier": "t_uint256",
                "typeString": "uint256"
            },
            "typeName": {
                "id": 4,
                "name": "uint",
                "nodeType": "ElementaryTypeName",
                "src": "69:4:0",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                }
            },
            "visibility": "internal"
        },
        {
            "constant": false,
            "id": 7,
            "mutability": "mutable",
            "name": "k",
            "nameLocation": "82:1:0",
            "nodeType": "VariableDeclaration",
            "src": "77:6:0",
            "stateVariable": true,
            "storageLocation": "default",
            "typeDescriptions": {
                "typeIdentifier": "t_uint256",
                "typeString": "uint256"
            },
            "typeName": {
                "id": 6,
                "name": "uint",
                "nodeType": "ElementaryTypeName",
                "src": "77:4:0",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                }
            },
            "visibility": "internal"
        }
    ],
    "src": "60:24:0"
},
"src": "45:121:0",
"stateMutability": "nonpayable",
"virtual": false,
"visibility": "private"
})";

inline const std::string mulmod_body =
  R"({
    "isTemplate": true,
    "templateFunctionID": "sol:@F@mulmod#",
    "body": {
        "id": 19,
        "nodeType": "Block",
        "src": "145:35:0",
        "statements": [
            {
                "expression": {
                    "commonType": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    },
                    "id": 17,
                    "isConstant": false,
                    "isLValue": false,
                    "isPure": false,
                    "lValueRequested": false,
                    "leftExpression": {
                        "components": [
                            {
                                "commonType": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                },
                                "id": 14,
                                "isConstant": false,
                                "isLValue": false,
                                "isPure": false,
                                "lValueRequested": false,
                                "leftExpression": {
                                    "id": 12,
                                    "name": "x",
                                    "nodeType": "Identifier",
                                    "overloadedDeclarations": [],
                                    "referencedDeclaration": 3,
                                    "src": "163:1:0",
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    }
                                },
                                "nodeType": "BinaryOperation",
                                "operator": "*",
                                "rightExpression": {
                                    "id": 13,
                                    "name": "y",
                                    "nodeType": "Identifier",
                                    "overloadedDeclarations": [],
                                    "referencedDeclaration": 5,
                                    "src": "167:1:0",
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    }
                                },
                                "src": "163:5:0",
                                "typeDescriptions": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                }
                            }
                        ],
                        "id": 15,
                        "isConstant": false,
                        "isInlineArray": false,
                        "isLValue": false,
                        "isPure": false,
                        "lValueRequested": false,
                        "nodeType": "TupleExpression",
                        "src": "162:7:0",
                        "typeDescriptions": {
                            "typeIdentifier": "t_uint256",
                            "typeString": "uint256"
                        }
                    },
                    "nodeType": "BinaryOperation",
                    "operator": "%",
                    "rightExpression": {
                        "id": 16,
                        "name": "k",
                        "nodeType": "Identifier",
                        "overloadedDeclarations": [],
                        "referencedDeclaration": 7,
                        "src": "172:1:0",
                        "typeDescriptions": {
                            "typeIdentifier": "t_uint256",
                            "typeString": "uint256"
                        }
                    },
                    "src": "162:11:0",
                    "typeDescriptions": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    }
                },
                "functionReturnParameters": 11,
                "id": 18,
                "nodeType": "Return",
                "src": "155:18:0"
            }
        ]
    },
    "id": 20,
    "implemented": true,
    "kind": "function",
    "modifiers": [],
    "name": "mulmod",
    "nameLocation": "91:6:0",
    "nodeType": "FunctionDefinition",
    "parameters": {
        "id": 8,
        "nodeType": "ParameterList",
        "parameters": [
            {
                "constant": false,
                "id": 3,
                "mutability": "mutable",
                "name": "x",
                "nameLocation": "103:1:0",
                "nodeType": "VariableDeclaration",
                "scope": 20,
                "src": "98:6:0",
                "stateVariable": false,
                "storageLocation": "default",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                },
                "typeName": {
                    "id": 2,
                    "name": "uint",
                    "nodeType": "ElementaryTypeName",
                    "src": "98:4:0",
                    "typeDescriptions": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    }
                },
                "visibility": "internal"
            },
            {
                "constant": false,
                "id": 5,
                "mutability": "mutable",
                "name": "y",
                "nameLocation": "111:1:0",
                "nodeType": "VariableDeclaration",
                "scope": 20,
                "src": "106:6:0",
                "stateVariable": false,
                "storageLocation": "default",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                },
                "typeName": {
                    "id": 4,
                    "name": "uint",
                    "nodeType": "ElementaryTypeName",
                    "src": "106:4:0",
                    "typeDescriptions": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    }
                },
                "visibility": "internal"
            },
            {
                "constant": false,
                "id": 7,
                "mutability": "mutable",
                "name": "k",
                "nameLocation": "119:1:0",
                "nodeType": "VariableDeclaration",
                "scope": 20,
                "src": "114:6:0",
                "stateVariable": false,
                "storageLocation": "default",
                "typeDescriptions": {
                    "typeIdentifier": "t_uint256",
                    "typeString": "uint256"
                },
                "typeName": {
                    "id": 6,
                    "name": "uint",
                    "nodeType": "ElementaryTypeName",
                    "src": "114:4:0",
                    "typeDescriptions": {
                        "typeIdentifier": "t_uint256",
                        "typeString": "uint256"
                    }
                },
                "visibility": "internal"
            }
        ],
        "src": "97:24:0"
    },
    "scope": 21,
    "src": "82:98:0",
    "stateMutability": "nonpayable",
    "virtual": false,
    "visibility": "private"
})";

// Library written in C

inline const std::string sol_header = R"(
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
)";

/* https://github.com/rxi/map */
inline const std::string sol_mapping = R"(
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
	/* char key[]; */
	/* char value[]; */
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
	/* If the implementation is changed to allow a non-power-of-2 bucket count,
	 * the line below should be changed to use mod instead of AND */
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
	/* Chain all nodes together */
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
	/* Find & replace existing node */
	next = map_getref(m, key);
	if (next)
	{
		memcpy((*next)->value, value, vsize);
		return 0;
	}
	/* Add new node */
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

// all content plus together
inline const std::string sol_library = sol_header + sol_mapping;

}; // namespace SolidityTemplate

#endif