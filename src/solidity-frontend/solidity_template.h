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
// #include <string>
)";

/*
  uint == uint256_t
  bytes == uint256_t
  bytes32 == uint256_t
  address == address_t
*/
const std::string sol_typedef = R"(
#if defined(__clang__)  // Ensure we are using Clang
    #if __clang_major__ >= 15
        #define BIGINT(bits) _BitInt(bits)
    #else
        #define BIGINT(bits) _ExtInt(bits)
    #endif
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
uint256_t gasleft()
{
__ESBMC_HIDE:;
  return nondet_uint();
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
  return address_t(x);
}
address_t ecrecover(uint256_t hash, unsigned int v, uint256_t r, uint256_t s)
{
__ESBMC_HIDE:;
  return address_t(hash);
}

uint256_t _pow(unsigned int base, unsigned int exp) {
__ESBMC_HIDE:;
  uint256_t result = 1;
  uint256_t b = base;

  while (exp > 0) {
    if (exp & 1)
      result *= b;
    b *= b;
    exp >>= 1;
  }

  return result;
}
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
char *u256toa(uint256_t value);
uint256_t str2int(const char *str);
uint256_t byte_concat(uint256_t x, uint256_t y)
{
__ESBMC_HIDE:;
  char *s1 = u256toa(x);
  char *s2 = u256toa(y);
  strncat(s1, s2, 256);
  return str2int(s1);
}
)";

const std::string sol_address = R"(
void _transfer(uint256_t ether, uint256_t balance)
{
__ESBMC_HIDE:;
  __ESBMC_assume(balance < ether);
}

bool _send(uint256_t ether, uint256_t balance)
{
__ESBMC_HIDE:;
  if(balance < ether)
    return false;
  return true;
}

bool _call()
{
__ESBMC_HIDE:;
  return nondet_bool();
}

bool _delegatecall()
{
__ESBMC_HIDE:;
  return nondet_bool();
}

bool _staticcall()
{
__ESBMC_HIDE:;
  return nondet_bool();
}

bool _callcodecall()
{
__ESBMC_HIDE:;
  return nondet_bool();
}

)";

const std::string sol_funcs = blockhash + gasleft + sol_abi + sol_math +
                              sol_string + sol_byte + sol_address;

/// data structure

/* https://github.com/rxi/map */
const std::string sol_mapping = R"(
struct NodeU
{
  uint256_t data : 256;
  struct NodeU *next;
};

struct NodeI
{
  int256_t data : 256;
  struct NodeI *next;
};

void insertAtEndU(struct NodeU **head, uint256_t data)
{
__ESBMC_HIDE:;
  struct NodeU *newNode = (struct NodeU *)malloc(sizeof(struct NodeU));
  newNode->data = data;
  newNode->next = NULL;
  if (*head == NULL)
  {
    *head = newNode;
    return;
  }
  struct NodeU *current = *head;
  while (current->next != NULL)
  {
    current = current->next;
  }
  current->next = newNode;
}

void insertAtEndI(struct NodeI **head, int256_t data)
{
__ESBMC_HIDE:;
  struct NodeI *newNode = (struct NodeI *)malloc(sizeof(struct NodeI));
  newNode->data = data;
  newNode->next = NULL;
  if (*head == NULL)
  {
    *head = newNode;
    return;
  }
  struct NodeI *current = *head;
  while (current->next != NULL)
  {
    current = current->next;
  }
  current->next = newNode;
}

int _ESBMC_uaddress(struct NodeU *head, uint256_t key)
{
__ESBMC_HIDE:;
  struct NodeU *current = head;
  int cnt = 0;
  while (current != NULL)
  {
    if (current->data == key)
      return cnt;
    cnt++;
    current = current->next;
  }
  insertAtEndU(&head, key);
  // temporary
  // if (cnt >= 50)
  //   assert(0);
  return cnt;
}

int _ESBMC_address(struct NodeI *head, int256_t key)
{
__ESBMC_HIDE:;
  struct NodeI *current = head;
  int cnt = 0;
  while (current != NULL)
  {
    if (current->data == key)
      return cnt;
    cnt++;
    current = current->next;
  }
  insertAtEndI(&head, key);
  // temporary
  // if (cnt >= 50)
  //   assert(0);
  return cnt;
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
unsigned int _ESBMC_get_array_length(void *array) {
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
    uint256_t ret = 0;
    while (*hex && ret >= (uint256_t)0)
    {
        ret = (ret << (uint256_t)4) | (uint256_t)hextable[*hex++];
    }
    return ret;
}
uint256_t str2int(const char *str)
{
__ESBMC_HIDE:;
    return hexdec(ASCIItoHEX(str));
}

// string assign
const char *empty_str = "";
void _str_assign(char **str1, const char *str2) {
__ESBMC_HIDE:;
    if(str1 != NULL)
      free(*str1);  
    if (str2 == NULL) {
      *str1 = NULL;  // Ensure str1 doesn't point to invalid memory
      return;
    }
    *str1 = (char *)malloc(strlen(str2) + 1);  
    
    strcpy(*str1, str2);  // force malloc success
}

)";

// get unique random address
const std::string sol_uqAddr = R"(
// compromise:
// - define a relatively large array
// static const unsigned int max_addr_obj_size = 50;
// static address_t sol_addr_array[max_addr_obj_size];
// static void *sol_obj_array[max_addr_obj_size];
// static char* sol_cname_array[max_addr_obj_size];
__attribute__((annotate("__ESBMC_inf_size"))) address_t sol_addr_array[1];
__attribute__((annotate("__ESBMC_inf_size"))) void *sol_obj_array[1];
__attribute__((annotate("__ESBMC_inf_size"))) char* sol_cname_array[1];
static unsigned sol_max_cnt = 0;

int _ESBMC_get_addr_array_idx(address_t tgt)
{
__ESBMC_HIDE:;
  for (unsigned int i = 0; i < sol_max_cnt; i++)
  {
    if ((address_t)sol_addr_array[i] == (address_t)tgt)
      return i;
  }
  return -1;
}
void *_ESBMC_get_obj(address_t addr)
{
__ESBMC_HIDE:;
  int idx = _ESBMC_get_addr_array_idx(addr);
  if (idx == -1)
    // this means it's not previously stored
    return NULL;
  else
    return sol_obj_array[idx];
}
void update_addr_obj(address_t addr, void *obj)
{
__ESBMC_HIDE:;
  __ESBMC_assume(obj != NULL);
  sol_addr_array[sol_max_cnt] = addr;
  sol_obj_array[sol_max_cnt] = obj;
  ++sol_max_cnt;
  // if (sol_max_cnt >= max_addr_obj_size)
  //   assert(0);
}
address_t _ESBMC_get_unique_address(void *obj)
{
__ESBMC_HIDE:;
  // __ESBMC_assume(obj != NULL);
  address_t tmp = (address_t)0;
  do
  {
    tmp = nondet_ulong() ; // ensure it's not address(0)
  } while (_ESBMC_get_addr_array_idx(tmp) != -1 && tmp != (address_t)0);
  // update_addr_obj(tmp, obj);
  return tmp;
}
void _ESBMC_set_cname_array(address_t _addr, char* cname)
{
__ESBMC_HIDE:;
  int tmp = _ESBMC_get_addr_array_idx(_addr);
  // assert(tmp != -1);
  sol_cname_array[tmp] = cname;
}
const char * _ESBMC_get_cname(address_t _addr)
{
__ESBMC_HIDE:;
  int tmp = _ESBMC_get_addr_array_idx(_addr);
  // assert(tmp != -1);
  return sol_cname_array[tmp];
}
bool _ESBMC_cmp_cname(const char* c_1, const char* c_2)
{
__ESBMC_HIDE:;
  return strcmp(c_1, c_2) == 0;
}

const char * _ESBMC_get_nondet_cont_name(const char *c_array[], unsigned int len)
{
__ESBMC_HIDE:;
unsigned int rand = nondet_uint() % len;
return c_array[rand];
}

uint256_t _ESBMC_update_balance(uint256_t balance, uint256_t val)
{
__ESBMC_HIDE:;
  val = val + (uint256_t)1;
  if(balance >= val)
    balance -= val;
  else
    balance = (uint256_t)0;
  return balance;
}
)";

// max/min value
const std::string sol_max_min = R"(
uint256_t _max(int bitwidth, bool is_signed) {
__ESBMC_HIDE:;
    if (is_signed) {
        return (uint256_t(1) << (bitwidth - 1)) - uint256_t(1); // 2^(N-1) - 1
    } else {
        return (uint256_t(1) << bitwidth) - uint256_t(1); // 2^N - 1
    }
}

int256_t _min(int bitwidth, bool is_signed) {
__ESBMC_HIDE:;
    if (is_signed) {
        return -(int256_t(1) << (bitwidth - 1)); // -2^(N-1)
    } else {
        return int256_t(0); // Min of unsigned is always 0
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

const std::string sol_ext_library =
  sol_itoa + sol_str2hex + sol_uqAddr + sol_max_min;

const std::string sol_c_library = "extern \"C\" {" + sol_typedef + sol_vars +
                                  sol_funcs + sol_mapping + sol_array +
                                  sol_ext_library + "}";

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

const std::string sol_signature = R"(
//TODO
)";

const std::string sol_cpp_library = sol_cpp_string + sol_signature;

// combination
const std::string sol_library =
  sol_header + sol_c_library; // + sol_cpp_library;

}; // namespace SolidityTemplate

#endif
