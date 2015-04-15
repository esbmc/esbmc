#include "irep2.h"
#include <stdarg.h>
#include <string.h>

#include "std_types.h"
#include "migrate.h"
#include "i2string.h"

#include <boost/algorithm/string.hpp>
#include <boost/static_assert.hpp>
#include <boost/functional/hash.hpp>

std::string
indent_str(unsigned int indent)
{
  return std::string(indent, ' ');
}

template <class T>
std::string
pretty_print_func(unsigned int indent, std::string ident, T obj)
{
  list_of_memberst memb = obj.tostring(indent+2);

  std::string indentstr = indent_str(indent);
  std::string exprstr = ident;

  for (list_of_memberst::const_iterator it = memb.begin(); it != memb.end();
       it++) {
    exprstr += "\n" + indentstr + "* " + it->first + " : " + it->second;
  }

  return exprstr;
}

/*************************** Base type2t definitions **************************/


static const char *type_names[] = {
  "bool",
  "empty",
  "symbol",
  "struct",
  "union",
  "code",
  "array",
  "pointer",
  "unsignedbv",
  "signedbv",
  "fixedbv",
  "string",
  "cpp_name"
};
// If this fires, you've added/removed a type id, and need to update the list
// above (which is ordered according to the enum list)
BOOST_STATIC_ASSERT(sizeof(type_names) ==
                    (type2t::end_type_id * sizeof(char *)));

std::string
get_type_id(const type2t &type)
{
  return std::string(type_names[type.type_id]);
}


type2t::type2t(type_ids id)
  : type_id(id),
    crc_val(0)
{
}

type2t::type2t(const type2t &ref)
  : type_id(ref.type_id),
    crc_val(ref.crc_val)
{
}

bool
type2t::operator==(const type2t &ref) const
{

  return cmpchecked(ref);
}

bool
type2t::operator<(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if (tmp < 0)
    return true;
  else if (tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

int
type2t::ltchecked(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if (tmp != 0)
    return tmp;

  return lt(ref);
}

bool
type2t::cmpchecked(const type2t &ref) const
{

  if (type_id == ref.type_id)
    return cmp(ref);

  return false;
}

int
type2t::lt(const type2t &ref) const
{

  if (type_id < ref.type_id)
    return -1;
  if (type_id > ref.type_id)
    return 1;
  return 0;
}

std::string
type2t::pretty(unsigned int indent) const
{

  return pretty_print_func<const type2t&>(indent, type_names[type_id], *this);
}

void
type2t::dump(void) const
{
  std::cout << pretty(0) << std::endl;
  return;
}

uint32_t
type2t::crc(void) const
{
  size_t seed = 0;
  do_crc(seed);
  return seed;
}

size_t
type2t::do_crc(size_t seed) const
{
  boost::hash_combine(seed, (uint8_t)type_id);
  return seed;
}

void
type2t::hash(crypto_hash &hash) const
{
  BOOST_STATIC_ASSERT(type2t::end_type_id < 256);
  uint8_t tid = type_id;
  hash.ingest(&tid, sizeof(tid));
  return;
}

unsigned int
bool_type2t::get_width(void) const
{
  // For the purpose of the byte representating memory model
  return 8;
}

unsigned int
bv_data::get_width(void) const
{
  return width;
}

unsigned int
array_type2t::get_width(void) const
{
  // Two edge cases: the array can have infinite size, or it can have a dynamic
  // size that's determined by the solver.
  if (size_is_infinite)
    throw new inf_sized_array_excp();

  if (array_size->expr_id != expr2t::constant_int_id)
    throw new dyn_sized_array_excp(array_size);

  // Otherwise, we can multiply the size of the subtype by the number of elements.
  unsigned int sub_width = subtype->get_width();

  const expr2t *elem_size = array_size.get();
  const constant_int2t *const_elem_size = dynamic_cast<const constant_int2t*>
                                                      (elem_size);
  assert(const_elem_size != NULL);
  unsigned long num_elems = const_elem_size->as_ulong();

  return num_elems * sub_width;
}

unsigned int
pointer_type2t::get_width(void) const
{
  return config.ansi_c.pointer_width;
}

unsigned int
empty_type2t::get_width(void) const
{
  throw new symbolic_type_excp();
}

unsigned int
symbol_type2t::get_width(void) const
{
  assert(0 && "Fetching width of symbol type - invalid operation");
}

unsigned int
cpp_name_type2t::get_width(void) const
{
  assert(0 && "Fetching width of cpp_name type - invalid operation");
}

unsigned int
struct_type2t::get_width(void) const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); it++)
    width += (*it)->get_width();

  return width;
}

unsigned int
union_type2t::get_width(void) const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); it++)
    width = std::max(width, (*it)->get_width());

  return width;
}

unsigned int
fixedbv_type2t::get_width(void) const
{
  return width;
}

unsigned int
code_data::get_width(void) const
{
  throw new symbolic_type_excp();
}

unsigned int
string_type2t::get_width(void) const
{
  return width * 8;
}

const std::vector<type2tc> &
struct_union_data::get_structure_members(void) const
{
  return members;
}

const std::vector<irep_idt> &
struct_union_data::get_structure_member_names(void) const
{
  return member_names;
}

const irep_idt &
struct_union_data::get_structure_name(void) const
{
  return name;
}

unsigned int
struct_union_data::get_component_number(const irep_idt &name) const
{

  unsigned int i = 0;
  forall_names(it, member_names) {
    if (*it == name)
      return i;
    i++;
  }

  std::cerr << "Looking up index of nonexistant member \"" << name
            << "\" in struct/union \"" << name << "\"" << std::endl;
  abort();
}

/*************************** Base expr2t definitions **************************/

expr2t::expr2t(const type2tc _type, expr_ids id)
  : std::enable_shared_from_this<expr2t>(), expr_id(id), type(_type), crc_val(0)
{
}

expr2t::expr2t(const expr2t &ref)
  : std::enable_shared_from_this<expr2t>(), expr_id(ref.expr_id),
    type(ref.type),
    crc_val(ref.crc_val)
{
}

bool
expr2t::operator==(const expr2t &ref) const
{
  if (!expr2t::cmp(ref))
    return false;

  return cmp(ref);
}

bool
expr2t::operator!=(const expr2t &ref) const
{
  return !(*this == ref);
}

bool
expr2t::operator<(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if (tmp < 0)
    return true;
  else if (tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

unsigned long
expr2t::depth(void) const
{
  unsigned long num_nodes = 0;

  for (unsigned int idx = 0; idx < get_num_sub_exprs(); idx++) {
    const expr2tc *e = get_sub_expr(idx);
    if (is_nil_expr(*e))
      continue;
    unsigned long tmp = (*e)->depth();
    num_nodes = std::max(num_nodes, tmp);
  }

  num_nodes++; // Count ourselves.
  return num_nodes;
}

unsigned long
expr2t::num_nodes(void) const
{
  unsigned long count = 0;

  for (unsigned int idx = 0; idx < get_num_sub_exprs(); idx++) {
    const expr2tc *e = get_sub_expr(idx);
    if (is_nil_expr(*e))
      continue;
    count += (*e)->num_nodes();
  }

  count++; // Count ourselves.
  return count;
}

int
expr2t::ltchecked(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if (tmp != 0)
    return tmp;

  return lt(ref);
}

bool
expr2t::cmp(const expr2t &ref) const
{
  if (expr_id != ref.expr_id)
    return false;

  if (type != ref.type)
    return false;

  return true;
}

int
expr2t::lt(const expr2t &ref) const
{
  if (expr_id < ref.expr_id)
    return -1;
  if (expr_id > ref.expr_id)
    return 1;

  return type->ltchecked(*ref.type.get());
}

uint32_t
expr2t::crc(void) const
{
  size_t seed = 0;
  return do_crc(seed);
}

size_t
expr2t::do_crc(size_t seed) const
{
  boost::hash_combine(seed, (uint8_t)expr_id);
  return type->do_crc(seed);
}

void
expr2t::hash(crypto_hash &hash) const
{
  BOOST_STATIC_ASSERT(expr2t::end_expr_id < 256);
  uint8_t eid = expr_id;
  hash.ingest(&eid, sizeof(eid));
  type->hash(hash);
  return;
}

expr2tc
expr2t::simplify(void) const
{

  // Try initial simplification
  expr2tc res = do_simplify();
  if (!is_nil_expr(res)) {
    // Woot, we simplified some of this. It may have _additional_ fields that
    // need to get simplified (member2ts in arrays for example), so invoke the
    // simplifier again, to hit those potential subfields.
    expr2tc res2 = res->simplify();

    // If we simplified even further, return res2; otherwise res.
    if (is_nil_expr(res2))
      return res;
    else
      return res2;
  }

  // Corner case! Don't even try to simplify address of's operands, might end up
  // taking the address of some /completely/ arbitary pice of data, by
  // simplifiying an index to its data, discarding the symbol.
  if (__builtin_expect((expr_id == address_of_id), 0)) // unlikely
    return expr2tc();

  // And overflows too. We don't wish an add to distribute itself, for example,
  // when we're trying to work out whether or not it's going to overflow.
  if (__builtin_expect((expr_id == overflow_id), 0))
    return expr2tc();

  // Try simplifying all the sub-operands.
  bool changed = false;
  std::list<expr2tc> newoperands;

  for (unsigned int idx = 0; idx < get_num_sub_exprs(); idx++) {
    const expr2tc *e = get_sub_expr(idx);
    expr2tc tmp;

    if (!is_nil_expr(*e)) {
      tmp = e->get()->simplify();
      if (!is_nil_expr(tmp))
        changed = true;
    }

    newoperands.push_back(tmp);
  }

  if (changed == false)
    // Second shot at simplification. For efficiency, a simplifier may be
    // holding something back until it's certain all its operands are
    // simplified. It's responsible for simplifying further if it's made that
    // call though.
    return do_simplify(true);

  // An operand has been changed; clone ourselves and update.
  expr2tc new_us = clone();
  std::list<expr2tc>::iterator it2 = newoperands.begin();
  Forall_operands2(it, idx, new_us) {
    if ((*it2) == NULL)
      ; // No change in operand;
    else
      *it = *it2; // Operand changed; overwrite with new one.
    it2++;
  }

  // Finally, attempt simplification again.
  expr2tc tmp = new_us->do_simplify(true);
  if (is_nil_expr(tmp))
    return new_us;
  else
    return tmp;
}

static const char *expr_names[] = {
  "constant_int",
  "constant_fixedbv",
  "constant_bool",
  "constant_string",
  "constant_struct",
  "constant_union",
  "constant_array",
  "constant_array_of",
  "symbol",
  "typecast",
  "if",
  "equality",
  "notequal",
  "lessthan",
  "greaterthan",
  "lessthanequal",
  "greaterthanequal",
  "not",
  "and",
  "or",
  "xor",
  "implies",
  "bitand",
  "bitor",
  "bitxor",
  "bitnand",
  "bitnor",
  "bitnxor",
  "bitnot",
  "lshr",
  "neg",
  "abs",
  "add",
  "sub",
  "mul",
  "div",
  "modulus",
  "shl",
  "ashr",
  "dynamic_object",
  "same_object",
  "pointer_offset",
  "pointer_object",
  "address_of",
  "byte_extract",
  "byte_update",
  "with",
  "member",
  "index",
  "zero_string",
  "zero_length_string",
  "isnan",
  "overflow",
  "overflow_cast",
  "overflow_neg",
  "unknown",
  "invalid",
  "NULL-object",
  "dereference",
  "valid_object",
  "deallocated_obj",
  "dynamic_size",
  "sideeffect",
  "code_block",
  "code_assign",
  "code_init",
  "code_decl",
  "code_printf",
  "code_expression",
  "code_return",
  "code_skip",
  "code_free",
  "code_goto",
  "object_descriptor",
  "code_function_call",
  "code_comma_id",
  "invalid_pointer",
  "buffer_size",
  "code_asm",
  "cpp_del_array",
  "cpp_delete",
  "cpp_catch",
  "cpp_throw",
  "cpp_throw_decl",
  "cpp_throw_decl_end",
  "isinf",
  "isnormal",
  "concat",
};
// If this fires, you've added/removed an expr id, and need to update the list
// above (which is ordered according to the enum list)
BOOST_STATIC_ASSERT(sizeof(expr_names) ==
                    (expr2t::end_expr_id * sizeof(char *)));

std::string
get_expr_id(const expr2t &expr)
{
  return std::string(expr_names[expr.expr_id]);
}

std::string
expr2t::pretty(unsigned int indent) const
{

  std::string ret = pretty_print_func<const expr2t&>(indent,
                                                     expr_names[expr_id],
                                                     *this);
  // Dump the type on the end.
  ret += std::string("\n") + indent_str(indent) + "  * type : "
         + type->pretty(indent+2);
  return ret;
}

void
expr2t::dump(void) const
{
  std::cout << pretty(0) << std::endl;
  return;
}

/**************************** Expression constructors *************************/

unsigned long
constant_int2t::as_ulong(void) const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  assert(!constant_value.is_negative());
  return constant_value.to_ulong();
}

long
constant_int2t::as_long(void) const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  return constant_value.to_long();
}

bool
constant_bool2t::is_true(void) const
{
  return constant_value;
}

bool
constant_bool2t::is_false(void) const
{
  return !constant_value;
}

std::string
symbol_data::get_symbol_name(void) const
{
  switch (rlevel) {
  case level0:
    return thename.as_string();
  case level1:
    return thename.as_string() + "@" + i2string(level1_num)
                               + "!" + i2string(thread_num);
  case level2:
    return thename.as_string() + "@" + i2string(level1_num)
                               + "!" + i2string(thread_num)
                               + "&" + i2string(node_num)
                               + "#" + i2string(level2_num);
  case level1_global:
    // Just return global name,
    return thename.as_string();
  case level2_global:
    // Global name with l2 details
    return thename.as_string() + "&" + i2string(node_num)
                               + "#" + i2string(level2_num);
  default:
    std::cerr << "Unrecognized renaming level enum" << std::endl;
    abort();
  }
}

expr2tc
constant_string2t::to_array(void) const
{
  std::vector<expr2tc> contents;
  unsigned int length = value.as_string().size(), i;

  type2tc type = type_pool.get_uint8();

  for (i = 0; i < length; i++) {
    constant_int2t *v = new constant_int2t(type, BigInt(value.as_string()[i]));
    expr2tc ptr(v);
    contents.push_back(ptr);
  }

  // Null terminator is implied.
  contents.push_back(expr2tc(new constant_int2t(type, BigInt(0))));

  unsignedbv_type2t *len_type = new unsignedbv_type2t(config.ansi_c.int_width);
  type2tc len_tp(len_type);
  constant_int2t *len_val = new constant_int2t(len_tp, BigInt(contents.size()));
  expr2tc len_val_ref(len_val);

  array_type2t *arr_type = new array_type2t(type, len_val_ref, false);
  type2tc arr_tp(arr_type);
  constant_array2t *a = new constant_array2t(arr_tp, contents);

  expr2tc final_val(a);
  return final_val;
}

const expr2tc &
object_descriptor2t::get_root_object(void) const
{
  const expr2tc *tmp = &object;

  do {
    if (is_member2t(*tmp))
      tmp = &to_member2t(*tmp).source_value;
    else if (is_index2t(*tmp))
      tmp = &to_index2t(*tmp).source_value;
    else
      return *tmp;
  } while (1);
}

type_poolt::type_poolt(void)
{
  // This space is deliberately left blank
}

type_poolt::type_poolt(bool yolo __attribute__((unused)))
{
  bool_type = type2tc(new bool_type2t());
  empty_type = type2tc(new empty_type2t());

  // Create some int types.
  type2tc ubv8(new unsignedbv_type2t(8));
  type2tc ubv16(new unsignedbv_type2t(16));
  type2tc ubv32(new unsignedbv_type2t(32));
  type2tc ubv64(new unsignedbv_type2t(64));
  type2tc sbv8(new signedbv_type2t(8));
  type2tc sbv16(new signedbv_type2t(16));
  type2tc sbv32(new signedbv_type2t(32));
  type2tc sbv64(new signedbv_type2t(64));

  unsignedbv_map[unsignedbv_typet(8)] = ubv8;
  unsignedbv_map[unsignedbv_typet(16)] = ubv16;
  unsignedbv_map[unsignedbv_typet(32)] = ubv32;
  unsignedbv_map[unsignedbv_typet(64)] = ubv64;
  signedbv_map[signedbv_typet(8)] = sbv8;
  signedbv_map[signedbv_typet(16)] = sbv16;
  signedbv_map[signedbv_typet(32)] = sbv32;
  signedbv_map[signedbv_typet(64)] = sbv64;

  uint8 = &unsignedbv_map[unsignedbv_typet(8)];
  uint16 = &unsignedbv_map[unsignedbv_typet(16)];
  uint32 = &unsignedbv_map[unsignedbv_typet(32)];
  uint64 = &unsignedbv_map[unsignedbv_typet(64)];
  int8 = &signedbv_map[signedbv_typet(8)];
  int16 = &signedbv_map[signedbv_typet(16)];
  int32 = &signedbv_map[signedbv_typet(32)];
  int64 = &signedbv_map[signedbv_typet(64)];

  return;
}

// XXX why did I disable this cache?
static const type2tc &
get_type_from_pool(const typet &val,
    std::map<typet, type2tc> &map __attribute__((unused)))
{
#if 0
  std::map<const typet, type2tc>::const_iterator it = map.find(val);
  if (it != map.end())
    return it->second;
#endif

  type2tc new_type;
  real_migrate_type(val, new_type);
#if 0
  map[val] = new_type;
  return map[val];
#endif
  return *(new type2tc(new_type));
}

const type2tc &
type_poolt::get_struct(const typet &val)
{
  return get_type_from_pool(val, struct_map);
}

const type2tc &
type_poolt::get_union(const typet &val)
{
  return get_type_from_pool(val, union_map);
}

const type2tc &
type_poolt::get_array(const typet &val)
{
  return get_type_from_pool(val, array_map);
}

const type2tc &
type_poolt::get_pointer(const typet &val)
{
  return get_type_from_pool(val, pointer_map);
}

const type2tc &
type_poolt::get_unsignedbv(const typet &val)
{
  return get_type_from_pool(val, unsignedbv_map);
}

const type2tc &
type_poolt::get_signedbv(const typet &val)
{
  return get_type_from_pool(val, signedbv_map);
}

const type2tc &
type_poolt::get_fixedbv(const typet &val)
{
  return get_type_from_pool(val, fixedbv_map);
}

const type2tc &
type_poolt::get_string(const typet &val)
{
  return get_type_from_pool(val, string_map);
}

const type2tc &
type_poolt::get_symbol(const typet &val)
{
  return get_type_from_pool(val, symbol_map);
}

const type2tc &
type_poolt::get_code(const typet &val)
{
  return get_type_from_pool(val, code_map);
}

const type2tc &
type_poolt::get_uint(unsigned int size)
{
  switch (size) {
  case 8:
    return get_uint8();
  case 16:
    return get_uint16();
  case 32:
    return get_uint32();
  case 64:
    return get_uint64();
  default:
    return get_unsignedbv(unsignedbv_typet(size));
  }
}

const type2tc &
type_poolt::get_int(unsigned int size)
{
  switch (size) {
  case 8:
    return get_int8();
  case 16:
    return get_int16();
  case 32:
    return get_int32();
  case 64:
    return get_int64();
  default:
    return get_signedbv(signedbv_typet(size));
  }
}

type_poolt type_pool;

// For CRCing to actually be accurate, expr/type ids mustn't overflow out of
// a byte. If this happens then a) there are too many exprs, and b) the expr
// crcing code has to change.
BOOST_STATIC_ASSERT(type2t::end_type_id <= 256);
BOOST_STATIC_ASSERT(expr2t::end_expr_id <= 256);

static inline __attribute__((always_inline)) std::string
type_to_string(const bool &thebool, int indent __attribute__((unused)))
{
  return (thebool) ? "true" : "false";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const sideeffect_data::allockind &data,
               int indent __attribute__((unused)))
{
  return (data == sideeffect_data::allockind::malloc) ? "malloc" :
         (data == sideeffect_data::allockind::cpp_new) ? "cpp_new" :
         (data == sideeffect_data::allockind::cpp_new_arr) ? "cpp_new_arr" :
         (data == sideeffect_data::allockind::nondet) ? "nondet" :
         (data == sideeffect_data::allockind::function_call) ? "function_call" :
         "unknown";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const unsigned int &theval, int indent __attribute__((unused)))
{
  char buffer[64];
  snprintf(buffer, 63, "%d", theval);
  return std::string(buffer);
}

static inline __attribute__((always_inline)) std::string
type_to_string(const symbol_data::renaming_level &theval,
               int indent __attribute__((unused)))
{
  switch (theval) {
  case symbol_data::level0:
    return "Level 0";
  case symbol_data::level1:
    return "Level 1";
  case symbol_data::level2:
    return "Level 2";
  case symbol_data::level1_global:
    return "Level 1 (global)";
  case symbol_data::level2_global:
    return "Level 2 (global)";
  default:
    std::cerr << "Unrecognized renaming level enum" << std::endl;
    abort();
  }
}

static inline __attribute__((always_inline)) std::string
type_to_string(const BigInt &theint, int indent __attribute__((unused)))
{
  char buffer[256], *buf;

  buf = theint.as_string(buffer, 256);
  return std::string(buf);
}

static inline __attribute__((always_inline)) std::string
type_to_string(const fixedbvt &theval, int indent __attribute__((unused)))
{
  return theval.to_ansi_c_string();
}

static inline __attribute__((always_inline)) std::string
type_to_string(const std::vector<expr2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  forall_exprs(it, theval) {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " + (*it)->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

static inline __attribute__((always_inline)) std::string
type_to_string(const std::vector<type2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  forall_types(it, theval) {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " + (*it)->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

static inline __attribute__((always_inline)) std::string
type_to_string(const std::vector<irep_idt> &theval,
               int indent __attribute__((unused)))
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  forall_names(it, theval) {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " + (*it).as_string() + "\n";
    i++;
  }

  return astring;
}

static inline __attribute__((always_inline)) std::string
type_to_string(const expr2tc &theval, int indent)
{

  if (theval.get() != NULL)
   return theval->pretty(indent + 2);
  return "";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const type2tc &theval, int indent)
{

  if (theval.get() != NULL)
    return theval->pretty(indent + 2);
  else
    return "";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const irep_idt &theval, int indent __attribute__((unused)))
{
  return theval.as_string();
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const bool &side1, const bool &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const unsigned int &side1, const unsigned int &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const sideeffect_data::allockind &side1,
            const sideeffect_data::allockind &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const symbol_data::renaming_level &side1,
            const symbol_data::renaming_level &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const BigInt &side1, const BigInt &side2)
{
  // BigInt has its own equality operator.
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const fixedbvt &side1, const fixedbvt &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const std::vector<expr2tc> &side1,
            const std::vector<expr2tc> &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const std::vector<type2tc> &side1,
            const std::vector<type2tc> &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const std::vector<irep_idt> &side1,
            const std::vector<irep_idt> &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // Catch null
  else if (side1.get() == NULL || side2.get() == NULL)
    return false;
  else
    return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const type2tc &side1, const type2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // both null ptr check
  if (side1.get() == NULL || side2.get() == NULL)
    return false; // One of them is null, the other isn't
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const irep_idt &side1, const irep_idt &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const type2t::type_ids &id __attribute__((unused)),
            const type2t::type_ids &id2 __attribute__((unused)))
{
  return true; // Dummy field comparison.
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const expr2t::expr_ids &id __attribute__((unused)),
            const expr2t::expr_ids &id2 __attribute__((unused)))
{
  return true; // Dummy field comparison.
}

static inline __attribute__((always_inline)) int
do_type_lt(const bool &side1, const bool &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const unsigned int &side1, const unsigned int &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const sideeffect_data::allockind &side1,
           const sideeffect_data::allockind &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const symbol_data::renaming_level &side1,
           const symbol_data::renaming_level &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const BigInt &side1, const BigInt &side2)
{
  // BigInt also has its own less than comparator.
  return side1.compare(side2);
}

static inline __attribute__((always_inline)) int
do_type_lt(const fixedbvt &side1, const fixedbvt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side1 > side2)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const std::vector<expr2tc> &side1, const std::vector<expr2tc> &side2)
{


  int tmp = 0;
  std::vector<expr2tc>::const_iterator it2 = side2.begin();
  forall_exprs(it, side1) {
    tmp = (*it)->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const std::vector<type2tc> &side1, const std::vector<type2tc> &side2)
{

  if (side1.size() < side2.size())
    return -1;
  else if (side1.size() > side2.size())
    return 1;

  int tmp = 0;
  std::vector<type2tc>::const_iterator it2 = side2.begin();
  forall_types(it, side1) {
    tmp = (*it)->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const std::vector<irep_idt> &side1,
           const std::vector<irep_idt> &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return 0; // Catch nulls
  else if (side1.get() == NULL)
    return -1;
  else if (side2.get() == NULL)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

static inline __attribute__((always_inline)) int
do_type_lt(const type2tc &side1, const type2tc &side2)
{
  if (*side1.get() == *side2.get())
    return 0; // Both may be null;
  else if (side1.get() == NULL)
    return -1;
  else if (side2.get() == NULL)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

static inline __attribute__((always_inline)) int
do_type_lt(const irep_idt &side1, const irep_idt &side2)
{
  if (side1 < side2)
    return -1;
  if (side2 < side1)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const type2t::type_ids &id __attribute__((unused)),
           const type2t::type_ids &id2 __attribute__((unused)))
{
  return 0; // Dummy field comparison
}

static inline __attribute__((always_inline)) int
do_type_lt(const expr2t::expr_ids &id __attribute__((unused)),
           const expr2t::expr_ids &id2 __attribute__((unused)))
{
  return 0; // Dummy field comparison
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const bool &thebool, size_t seed)
{

  boost::hash_combine(seed, thebool);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const bool &thebool, crypto_hash &hash)
{

  if (thebool) {
    uint8_t tval = 1;
    hash.ingest(&tval, sizeof(tval));
  } else {
    uint8_t tval = 0;
    hash.ingest(&tval, sizeof(tval));
  }
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const unsigned int &theval, size_t seed)
{

  boost::hash_combine(seed, theval);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const unsigned int &theval, crypto_hash &hash)
{

  hash.ingest((void*)&theval, sizeof(theval));
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const sideeffect_data::allockind &theval, size_t seed)
{

  boost::hash_combine(seed, (uint8_t)theval);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const sideeffect_data::allockind &theval, crypto_hash &hash)
{

  hash.ingest((void*)&theval, sizeof(theval));
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const symbol_data::renaming_level &theval, size_t seed)
{

  boost::hash_combine(seed, (uint8_t)theval);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const symbol_data::renaming_level &theval, crypto_hash &hash)
{

  hash.ingest((void*)&theval, sizeof(theval));
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const BigInt &theint, size_t seed)
{
  unsigned char buffer[256];

  if (theint.dump(buffer, sizeof(buffer))) {
    // Zero has no data in bigints.
    if (theint.is_zero()) {
      boost::hash_combine(seed, 0);
    } else {
      unsigned int thelen = theint.get_len();
      thelen *= 4; // words -> bytes
      unsigned int start = 256 - thelen;
      for (unsigned int i = 0; i < thelen; i++)
        boost::hash_combine(seed, buffer[start + i]);
    }
  } else {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const BigInt &theint, crypto_hash &hash)
{
  unsigned char buffer[256];

  if (theint.dump(buffer, sizeof(buffer))) {
    // Zero has no data in bigints.
    if (theint.is_zero()) {
      uint8_t val = 0;
      hash.ingest(&val, sizeof(val));
    } else {
      hash.ingest(buffer, theint.get_len());
    }
  } else {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const fixedbvt &theval, size_t seed)
{

  return do_type_crc(theval.get_value(), seed);
}

static inline __attribute__((always_inline)) void
do_type_hash(const fixedbvt &theval, crypto_hash &hash)
{

  do_type_hash(theval.to_integer(), hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<expr2tc> &theval, size_t seed)
{
  forall_exprs(it, theval)
    (*it)->do_crc(seed);

  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash)
{
  forall_exprs(it, theval)
    (*it)->hash(hash);
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<type2tc> &theval, size_t seed)
{
  forall_types(it, theval)
    (*it)->do_crc(seed);

  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash)
{
  forall_types(it, theval)
    (*it)->hash(hash);
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<irep_idt> &theval, size_t seed)
{
  forall_names(it, theval)
    boost::hash_combine(seed, (*it).as_string());
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash)
{
  forall_names(it, theval)
    hash.ingest((void*)(*it).as_string().c_str(), (*it).as_string().size());
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<unsigned int> &theval, size_t seed)
{
  for (std::vector<unsigned int>::const_iterator it = theval.begin();
       it != theval.end(); it++)
    boost::hash_combine(seed, *it);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<unsigned int> &theval, crypto_hash &hash)
{
  for (std::vector<unsigned int>::const_iterator it = theval.begin();
       it != theval.end(); it++)
    hash.ingest((void*)&(*it), sizeof(unsigned int));
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const expr2tc &theval, size_t seed)
{

  if (theval.get() != NULL)
    return theval->do_crc(seed);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const expr2tc &theval, crypto_hash &hash)
{

  if (theval.get() != NULL)
    theval->hash(hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const type2tc &theval, size_t seed)
{

  if (theval.get() != NULL)
    return theval->do_crc(seed);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const type2tc &theval, crypto_hash &hash)
{

  if (theval.get() != NULL)
    theval->hash(hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const irep_idt &theval, size_t seed)
{

  boost::hash_combine(seed, theval.as_string());
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const irep_idt &theval, crypto_hash &hash)
{

  hash.ingest((void*)theval.as_string().c_str(), theval.as_string().size());
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const type2t::type_ids &i __attribute__((unused)), size_t seed)
{
  return seed; // Dummy field crc
}

static inline __attribute__((always_inline)) void
do_type_hash(const type2t::type_ids &i __attribute__((unused)),
             crypto_hash &hash __attribute__((unused)))
{
  return; // Dummy field crc
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const expr2t::expr_ids &i __attribute__((unused)), size_t seed)
{
  return seed; // Dummy field crc
}

static inline __attribute__((always_inline)) void
do_type_hash(const expr2t::expr_ids &i __attribute__((unused)),
             crypto_hash &hash __attribute__((unused)))
{
  return; // Dummy field crc
}

static inline __attribute__((always_inline)) void do_type_list_operands(const symbol_data::renaming_level &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const type2tc &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const bool &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const sideeffect_data::allockind  &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const unsigned int &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const BigInt &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const fixedbvt &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const dstring &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const expr2t::expr_ids &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const std::vector<irep_idt> &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }

static inline __attribute__((always_inline)) void do_type_list_operands(symbol_data::renaming_level &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(type2tc &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(bool &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(sideeffect_data::allockind &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(unsigned int &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(BigInt &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(fixedbvt &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(dstring &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(const expr2t::expr_ids &theval __attribute__((unused)), std::list< expr2tc*> &inp __attribute__((unused))) { return; }
static inline __attribute__((always_inline)) void do_type_list_operands(std::vector<irep_idt> &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }

static inline __attribute__((always_inline)) void
do_type_list_operands(expr2tc &theval, std::list<expr2tc*> &inp)
{
  if (is_nil_expr(theval))
    return;

  inp.push_back(&theval);
}

static inline __attribute__((always_inline)) void
do_type_list_operands(std::vector<expr2tc> &theval, std::list<expr2tc*> &inp)
{
  for (std::vector<expr2tc>::iterator it = theval.begin(); it != theval.end();
       it++) {
    if (!is_nil_expr(*it))
      inp.push_back(&(*it));
  }
}

static inline __attribute__((always_inline)) void
do_type_list_operands(const expr2tc &theval, std::list<const expr2tc *> &inp)
{
  if (is_nil_expr(theval))
    return;

  inp.push_back(&theval);
}

static inline __attribute__((always_inline)) void
do_type_list_operands(const std::vector<expr2tc> &theval,
                      std::list<const expr2tc *> &inp)
{
  for (std::vector<expr2tc>::const_iterator it = theval.begin();
       it != theval.end(); it++) {
    if (!is_nil_expr(*it))
      inp.push_back(&(*it));
  }
}

template <typename T>
void
do_type2string(const T &thething, unsigned int idx,
               std::string (&names)[esbmct::num_type_fields],
               list_of_memberst &vec, unsigned int indent)
{
  vec.push_back(member_entryt(names[idx], type_to_string(thething, indent)));
}

template <>
void
do_type2string<type2t::type_ids>(
    const type2t::type_ids &thething __attribute__((unused)),
    unsigned int idx __attribute__((unused)),
    std::string (&names)[esbmct::num_type_fields] __attribute__((unused)),
    list_of_memberst &vec __attribute__((unused)),
    unsigned int indent __attribute__((unused)))
{
  // Do nothing; this is a dummy member.
}

template <>
void
do_type2string<const expr2t::expr_ids>(
    const expr2t::expr_ids &thething __attribute__((unused)),
    unsigned int idx __attribute__((unused)),
    std::string (&names)[esbmct::num_type_fields] __attribute__((unused)),
    list_of_memberst &vec __attribute__((unused)),
    unsigned int indent __attribute__((unused)))
{
  // Do nothing; this is a dummy member.
}

template <class T>
bool
do_get_sub_expr(const T &item __attribute__((unused)),
                unsigned int idx __attribute__((unused)),
                unsigned int &it __attribute__((unused)),
                const expr2tc *&ptr __attribute__((unused)))
{
  return false;
}

template <>
bool
do_get_sub_expr<expr2tc>(const expr2tc &item, unsigned int idx,
                               unsigned int &it, const expr2tc *&ptr)
{
  if (idx == it) {
    ptr = &item;
    return true;
  } else {
    it++;
    return false;
  }
}

template <>
bool
do_get_sub_expr<std::vector<expr2tc>>(const std::vector<expr2tc> &item,
                                      unsigned int idx, unsigned int &it,
                                      const expr2tc *&ptr)
{
  if (idx < it + item.size()) {
    ptr = &item[idx - it];
    return true;
  } else {
    it += item.size();
    return false;
  }
}

// Non-const versions of the above.

template <class T>
bool
do_get_sub_expr(T &item __attribute__((unused)),
                unsigned int idx __attribute__((unused)),
                unsigned int &it __attribute__((unused)),
                expr2tc *&ptr __attribute__((unused)))
{
  return false;
}

template <>
bool
do_get_sub_expr<expr2tc>(expr2tc &item, unsigned int idx, unsigned int &it,
                         expr2tc *&ptr)
{
  if (idx == it) {
    ptr = &item;
    return true;
  } else {
    it++;
    return false;
  }
}

template <>
bool
do_get_sub_expr<std::vector<expr2tc>>(std::vector<expr2tc> &item,
                                      unsigned int idx, unsigned int &it,
                                      expr2tc *&ptr)
{
  if (idx < it + item.size()) {
    ptr = &item[idx - it];
    return true;
  } else {
    it += item.size();
    return false;
  }
}

template <class T>
unsigned int
do_count_sub_exprs(T &item __attribute__((unused)))
{
  return 0;
}

template <>
unsigned int
do_count_sub_exprs<const expr2tc>(const expr2tc &item __attribute__((unused)))
{
  return 1;
}

template <>
unsigned int
do_count_sub_exprs<const std::vector<expr2tc>>(const std::vector<expr2tc> &item)
{
  return item.size();
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
expr2tc
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
expr2t *
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::clone_raw(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return new_obj;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
list_of_memberst
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::tostring(unsigned int indent) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  list_of_memberst thevector;
  do_type2string<field1_type>(derived_this->*field1_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field2_type>(derived_this->*field2_ptr, 1,
                              derived_this->field_names, thevector, indent);
  do_type2string<field3_type>(derived_this->*field3_ptr, 2,
                              derived_this->field_names, thevector, indent);
  do_type2string<field4_type>(derived_this->*field4_ptr, 3,
                              derived_this->field_names, thevector, indent);
  do_type2string<field5_type>(derived_this->*field5_ptr, 4,
                              derived_this->field_names, thevector, indent);
  do_type2string<field6_type>(derived_this->*field6_ptr, 5,
                              derived_this->field_names, thevector, indent);
  return thevector;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
bool
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::cmp(const expr2t &ref)const
{
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  if (!do_type_cmp(derived_this->*field1_ptr, ref2->*field1_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field2_ptr, ref2->*field2_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field3_ptr, ref2->*field3_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field4_ptr, ref2->*field4_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field5_ptr, ref2->*field5_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field6_ptr, ref2->*field6_ptr))
    return false;

  return true;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
int
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::lt(const expr2t &ref)const
{
  int tmp;
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  tmp = do_type_lt(derived_this->*field1_ptr, ref2->*field1_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field2_ptr, ref2->*field2_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field3_ptr, ref2->*field3_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field4_ptr, ref2->*field4_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field5_ptr, ref2->*field5_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field6_ptr, ref2->*field6_ptr);
  if (tmp != 0)
    return tmp;

  return tmp;
}

typedef std::size_t lolnoop;
inline std::size_t
hash_value(lolnoop val)
{
  return val;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
size_t
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::do_crc
          (size_t seed) const
{
  const derived *derived_this = static_cast<const derived*>(this);

  if (this->crc_val != 0) {
    boost::hash_combine(seed, (lolnoop)this->crc_val);
    return seed;
  }



  // Starting from 0, pass a crc value through all the sub-fields of this
  // expression. Store it into crc_val. Don't allow the input seed to affect
  // this calculation, as the crc value needs to uniquely identify _this_
  // expression.
  assert(this->crc_val == 0);
  size_t tmp = derived_this->expr2t::do_crc(0);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field1_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field2_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field3_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field4_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field5_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field6_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);

  // Finally, combine the crc of this expr with the input seed, and return
  boost::hash_combine(seed, (lolnoop)this->crc_val);
  return seed;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
void
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::hash
          (crypto_hash &hash) const
{
  const derived *derived_this = static_cast<const derived*>(this);

  derived_this->expr2t::hash(hash);
  do_type_hash(derived_this->*field1_ptr, hash);
  do_type_hash(derived_this->*field2_ptr, hash);
  do_type_hash(derived_this->*field3_ptr, hash);
  do_type_hash(derived_this->*field4_ptr, hash);
  do_type_hash(derived_this->*field5_ptr, hash);
  do_type_hash(derived_this->*field6_ptr, hash);
  return;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
void
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::list_operands
          (std::list<const expr2tc *> &inp) const
{
  const derived *derived_this = static_cast<const derived*>(this);

  do_type_list_operands(derived_this->*field1_ptr, inp);
  do_type_list_operands(derived_this->*field2_ptr, inp);
  do_type_list_operands(derived_this->*field3_ptr, inp);
  do_type_list_operands(derived_this->*field4_ptr, inp);
  do_type_list_operands(derived_this->*field5_ptr, inp);
  do_type_list_operands(derived_this->*field6_ptr, inp);
  return;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
const expr2tc *
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::get_sub_expr
          (unsigned int idx) const
{
  unsigned int it = 0;
  const expr2tc *ptr;
  const derived *derived_this = static_cast<const derived*>(this);

  if (do_get_sub_expr(derived_this->*field1_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field2_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field3_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field4_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field5_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field6_ptr, idx, it, ptr))
    return ptr;
  return NULL;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
expr2tc *
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::get_sub_expr_nc
          (unsigned int idx)
{
  unsigned int it = 0;
  expr2tc *ptr;
  derived *derived_this = static_cast<derived*>(this);

  if (do_get_sub_expr(derived_this->*field1_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field2_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field3_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field4_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field5_ptr, idx, it, ptr))
    return ptr;
  if (do_get_sub_expr(derived_this->*field6_ptr, idx, it, ptr))
    return ptr;
  return NULL;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
unsigned int
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::get_num_sub_exprs() const
{
  unsigned int num = 0;
  const derived *derived_this = static_cast<const derived*>(this);

  num += do_count_sub_exprs(derived_this->*field1_ptr);
  num += do_count_sub_exprs(derived_this->*field2_ptr);
  num += do_count_sub_exprs(derived_this->*field3_ptr);
  num += do_count_sub_exprs(derived_this->*field4_ptr);
  num += do_count_sub_exprs(derived_this->*field5_ptr);
  num += do_count_sub_exprs(derived_this->*field6_ptr);
  return num;
}

template <class derived, class subclass,
typename field1_type, class field1_class, field1_type field1_class::*field1_ptr,
typename field2_type, class field2_class, field2_type field2_class::*field2_ptr,
typename field3_type, class field3_class, field3_type field3_class::*field3_ptr,
typename field4_type, class field4_class, field4_type field4_class::*field4_ptr,
typename field5_type, class field5_class, field5_type field5_class::*field5_ptr,
typename field6_type, class field6_class, field6_type field6_class::*field6_ptr>
void
esbmct::expr_methods<derived, subclass,
  field1_type, field1_class, field1_ptr,
  field2_type, field2_class, field2_ptr,
  field3_type, field3_class, field3_ptr,
  field4_type, field4_class, field4_ptr,
  field5_type, field5_class, field5_ptr,
  field6_type, field6_class, field6_ptr>
  ::list_operands
          (std::list<expr2tc*> &inp)
{
  derived *derived_this = static_cast<derived*>(this);

  do_type_list_operands(derived_this->*field1_ptr, inp);
  do_type_list_operands(derived_this->*field2_ptr, inp);
  do_type_list_operands(derived_this->*field3_ptr, inp);
  do_type_list_operands(derived_this->*field4_ptr, inp);
  do_type_list_operands(derived_this->*field5_ptr, inp);
  do_type_list_operands(derived_this->*field6_ptr, inp);
  return;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr,
  class field5_type, class field5_class, field5_type field5_class::*field5_ptr,
  class field6_type, class field6_class, field6_type field6_class::*field6_ptr>
type2tc
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr,
                                        field5_type, field5_class, field5_ptr,
                                        field6_type, field6_class, field6_ptr>
      ::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return type2tc(new_obj);
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr,
  class field5_type, class field5_class, field5_type field5_class::*field5_ptr,
  class field6_type, class field6_class, field6_type field6_class::*field6_ptr>
list_of_memberst
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr,
                                        field5_type, field5_class, field5_ptr,
                                        field6_type, field6_class, field6_ptr>
      ::tostring(unsigned int indent) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  list_of_memberst thevector;
  do_type2string<field1_type>(derived_this->*field1_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field2_type>(derived_this->*field2_ptr, 1,
                              derived_this->field_names, thevector, indent);
  do_type2string<field3_type>(derived_this->*field3_ptr, 2,
                              derived_this->field_names, thevector, indent);
  do_type2string<field4_type>(derived_this->*field4_ptr, 3,
                              derived_this->field_names, thevector, indent);
  do_type2string<field5_type>(derived_this->*field5_ptr, 4,
                              derived_this->field_names, thevector, indent);
  do_type2string<field6_type>(derived_this->*field6_ptr, 5,
                              derived_this->field_names, thevector, indent);
  return thevector;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr,
  class field5_type, class field5_class, field5_type field5_class::*field5_ptr,
  class field6_type, class field6_class, field6_type field6_class::*field6_ptr>
bool
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr,
                                        field5_type, field5_class, field5_ptr,
                                        field6_type, field6_class, field6_ptr>
      ::cmp(const type2t &ref) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  if (!do_type_cmp(derived_this->*field1_ptr, ref2->*field1_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field2_ptr, ref2->*field2_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field3_ptr, ref2->*field3_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field4_ptr, ref2->*field4_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field5_ptr, ref2->*field5_ptr))
    return false;

  if (!do_type_cmp(derived_this->*field6_ptr, ref2->*field6_ptr))
    return false;

  return true;
}


template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr,
  class field5_type, class field5_class, field5_type field5_class::*field5_ptr,
  class field6_type, class field6_class, field6_type field6_class::*field6_ptr>
int
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr,
                                        field5_type, field5_class, field5_ptr,
                                        field6_type, field6_class, field6_ptr>
      ::lt(const type2t &ref)const
{
  int tmp;
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  tmp = do_type_lt(derived_this->*field1_ptr, ref2->*field1_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field2_ptr, ref2->*field2_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field3_ptr, ref2->*field3_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field4_ptr, ref2->*field4_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field5_ptr, ref2->*field5_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt(derived_this->*field6_ptr, ref2->*field6_ptr);
  if (tmp != 0)
    return tmp;

  return tmp;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr,
  class field5_type, class field5_class, field5_type field5_class::*field5_ptr,
  class field6_type, class field6_class, field6_type field6_class::*field6_ptr>
size_t
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr,
                                        field5_type, field5_class, field5_ptr,
                                        field6_type, field6_class, field6_ptr>
      ::do_crc (size_t seed) const
{

  const derived *derived_this = static_cast<const derived*>(this);

  if (this->crc_val != 0) {
    boost::hash_combine(seed, (lolnoop)this->crc_val);
    return seed;
  }

  // Starting from 0, pass a crc value through all the sub-fields of this
  // expression. Store it into crc_val. Don't allow the input seed to affect
  // this calculation, as the crc value needs to uniquely identify _this_
  // expression.
  assert(this->crc_val == 0);
  size_t tmp = derived_this->type2t::do_crc(0);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field1_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field2_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field3_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field4_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field5_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);
  tmp = do_type_crc(derived_this->*field6_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);

  // Finally, combine the crc of this expr with the input seed, and return
  boost::hash_combine(seed, (lolnoop)this->crc_val);
  return seed;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr,
  class field5_type, class field5_class, field5_type field5_class::*field5_ptr,
  class field6_type, class field6_class, field6_type field6_class::*field6_ptr>
void
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr,
                                        field5_type, field5_class, field5_ptr,
                                        field6_type, field6_class, field6_ptr>
      ::hash(crypto_hash &hash) const
{

  const derived *derived_this = static_cast<const derived*>(this);

  derived_this->type2t::hash(hash);
  do_type_hash(derived_this->*field1_ptr, hash);
  do_type_hash(derived_this->*field2_ptr, hash);
  do_type_hash(derived_this->*field3_ptr, hash);
  do_type_hash(derived_this->*field4_ptr, hash);
  do_type_hash(derived_this->*field5_ptr, hash);
  do_type_hash(derived_this->*field6_ptr, hash);
  return;
}

const expr2tc true_expr;
const expr2tc false_expr;

const constant_int2tc zero_u32;
const constant_int2tc one_u32;
const constant_int2tc zero_32;
const constant_int2tc one_32;
const constant_int2tc zero_u64;
const constant_int2tc one_u64;
const constant_int2tc zero_64;
const constant_int2tc one_64;

const constant_int2tc zero_ulong;
const constant_int2tc one_ulong;
const constant_int2tc zero_long;
const constant_int2tc one_long;

// More avoidance of static initialization order fiasco
void
init_expr_constants(void)
{
  const_cast<expr2tc&>(true_expr) = expr2tc(new constant_bool2t(true));
  const_cast<expr2tc&>(false_expr) = expr2tc(new constant_bool2t(false));

  const_cast<constant_int2tc&>(zero_u32)
    = constant_int2tc(type_pool.get_uint(32), BigInt(0));
  const_cast<constant_int2tc&>(one_u32)
    = constant_int2tc(type_pool.get_uint(32), BigInt(1));
  const_cast<constant_int2tc&>(zero_32)
    = constant_int2tc(type_pool.get_int(32), BigInt(0));
  const_cast<constant_int2tc&>(one_32)
    = constant_int2tc(type_pool.get_int(32), BigInt(1));

  const_cast<constant_int2tc&>(zero_u64)
    = constant_int2tc(type_pool.get_uint(64), BigInt(0));
  const_cast<constant_int2tc&>(one_u64)
    = constant_int2tc(type_pool.get_uint(64), BigInt(1));
  const_cast<constant_int2tc&>(zero_64)
    = constant_int2tc(type_pool.get_int(64), BigInt(0));
  const_cast<constant_int2tc&>(one_64)
    = constant_int2tc(type_pool.get_int(64), BigInt(1));

}

std::string bool_type2t::field_names [esbmct::num_type_fields]  = {"","","","", ""};
std::string empty_type2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string symbol_type2t::field_names [esbmct::num_type_fields]  =
{ "symbol_name", "", "", "", ""};
std::string struct_type2t::field_names [esbmct::num_type_fields]  =
{ "members", "member_names", "typename", "", ""};
std::string union_type2t::field_names [esbmct::num_type_fields]  =
{ "members", "member_names", "typename", "", ""};
std::string unsignedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", "", ""};
std::string signedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", "", ""};
std::string code_type2t::field_names [esbmct::num_type_fields]  =
{ "arguments", "ret_type", "argument_names", "ellipsis", ""};
std::string array_type2t::field_names [esbmct::num_type_fields]  =
{ "subtype", "array_size", "size_is_infinite", "", ""};
std::string pointer_type2t::field_names [esbmct::num_type_fields]  =
{ "subtype", "", "", "", ""};
std::string fixedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "integer_bits", "", "", ""};
std::string string_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", "", ""};
std::string cpp_name_type2t::field_names [esbmct::num_type_fields]  =
{ "name", "template args", "", "", ""};

// Exprs

std::string constant_int2t::field_names [esbmct::num_type_fields]  =
{ "constant_value", "", "", "", ""};
std::string constant_fixedbv2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string constant_struct2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", "", ""};
std::string constant_union2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", "", ""};
std::string constant_bool2t::field_names [esbmct::num_type_fields]  =
{ "constant_value", "", "", "", ""};
std::string constant_array2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", "", ""};
std::string constant_array_of2t::field_names [esbmct::num_type_fields]  =
{ "initializer", "", "", "", ""};
std::string constant_string2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string symbol2t::field_names [esbmct::num_type_fields]  =
{ "name", "renamelev", "level1_num", "level2_num", "thread_num", "node_num"};
std::string typecast2t::field_names [esbmct::num_type_fields]  =
{ "from", "", "", "", ""};
std::string if2t::field_names [esbmct::num_type_fields]  =
{ "cond", "true_value", "false_value", "", ""};
std::string equality2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string notequal2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string lessthan2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string greaterthan2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string lessthanequal2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string greaterthanequal2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string not2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string and2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string or2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string xor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string implies2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitand2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitxor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnand2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnxor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string lshr2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnot2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string neg2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string abs2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string add2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string sub2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string mul2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string div2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string modulus2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string shl2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string ashr2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string same_object2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string pointer_offset2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string pointer_object2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string address_of2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string byte_extract2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "source_offset", "big_endian", "", ""};
std::string byte_update2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "source_offset", "update_value", "big_endian", ""};
std::string with2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "update_field", "update_value", "", ""};
std::string member2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "member_name", "", "", ""};
std::string index2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "index", "", "", ""};
std::string zero_string2t::field_names [esbmct::num_type_fields]  =
{ "string", "", "", "", ""};
std::string zero_length_string2t::field_names [esbmct::num_type_fields]  =
{ "string", "", "", "", ""};
std::string isnan2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string overflow2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string overflow_cast2t::field_names [esbmct::num_type_fields]  =
{ "operand", "bits", "", "", ""};
std::string overflow_neg2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string unknown2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string invalid2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string null_object2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string dynamic_object2t::field_names [esbmct::num_type_fields]  =
{ "instance", "invalid", "unknown", "", ""};
std::string dereference2t::field_names [esbmct::num_type_fields]  =
{ "pointer", "", "", "", ""};
std::string valid_object2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string deallocated_obj2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string dynamic_size2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string sideeffect2t::field_names [esbmct::num_type_fields]  =
{ "operand", "size", "arguments", "alloctype", "kind"};
std::string code_block2t::field_names [esbmct::num_type_fields]  =
{ "operands", "", "", "", ""};
std::string code_assign2t::field_names [esbmct::num_type_fields]  =
{ "target", "source", "", "", ""};
std::string code_init2t::field_names [esbmct::num_type_fields]  =
{ "target", "source", "", "", ""};
std::string code_decl2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_printf2t::field_names [esbmct::num_type_fields]  =
{ "operands", "", "", "", ""};
std::string code_expression2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string code_return2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string code_skip2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string code_free2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string code_goto2t::field_names [esbmct::num_type_fields]  =
{ "target", "", "", "", ""};
std::string object_descriptor2t::field_names [esbmct::num_type_fields]  =
{ "object", "offset", "alignment", "", ""};
std::string code_function_call2t::field_names [esbmct::num_type_fields]  =
{ "return", "function", "operands", "", ""};
std::string code_comma2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string invalid_pointer2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string buffer_size2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_asm2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_cpp_del_array2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_cpp_delete2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_cpp_catch2t::field_names [esbmct::num_type_fields]  =
{ "exception_list", "", "", "", ""};
std::string code_cpp_throw2t::field_names [esbmct::num_type_fields]  =
{ "operand", "exception_list", "", "", ""};
std::string code_cpp_throw_decl2t::field_names [esbmct::num_type_fields]  =
{ "exception_list", "", "", "", ""};
std::string code_cpp_throw_decl_end2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string isinf2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string isnormal2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string concat2t::field_names [esbmct::num_type_fields]  =
{ "forward", "aft", "", "", ""};

// Explicit template instanciations

template class esbmct::type_methods<bool_type2t, type2t>;
template class esbmct::type_methods<empty_type2t, type2t>;
template class esbmct::type_methods<symbol_type2t, symbol_type_data, irep_idt,
               symbol_type_data, &symbol_type_data::symbol_name>;
template class esbmct::type_methods<struct_type2t, struct_union_data,
    std::vector<type2tc>, struct_union_data, &struct_union_data::members,
    std::vector<irep_idt>, struct_union_data, &struct_union_data::member_names,
    irep_idt, struct_union_data, &struct_union_data::name>;
template class esbmct::type_methods<union_type2t, struct_union_data,
    std::vector<type2tc>, struct_union_data, &struct_union_data::members,
    std::vector<irep_idt>, struct_union_data, &struct_union_data::member_names,
    irep_idt, struct_union_data, &struct_union_data::name>;
template class esbmct::type_methods<unsignedbv_type2t, bv_data,
    unsigned int, bv_data, &bv_data::width>;
template class esbmct::type_methods<signedbv_type2t, bv_data,
    unsigned int, bv_data, &bv_data::width>;
template class esbmct::type_methods<code_type2t, code_data,
    std::vector<type2tc>, code_data, &code_data::arguments,
    type2tc, code_data, &code_data::ret_type,
    std::vector<irep_idt>, code_data, &code_data::argument_names,
    bool, code_data, &code_data::ellipsis>;
template class esbmct::type_methods<array_type2t, array_data,
    type2tc, array_data, &array_data::subtype,
    expr2tc, array_data, &array_data::array_size,
    bool, array_data, &array_data::size_is_infinite>;
template class esbmct::type_methods<pointer_type2t, pointer_data,
    type2tc, pointer_data, &pointer_data::subtype>;
template class esbmct::type_methods<fixedbv_type2t, fixedbv_data,
    unsigned int, fixedbv_data, &fixedbv_data::width,
    unsigned int, fixedbv_data, &fixedbv_data::integer_bits>;
template class esbmct::type_methods<string_type2t, string_data,
    unsigned int, string_data, &string_data::width>;
template class esbmct::type_methods<cpp_name_type2t, cpp_name_data,
    irep_idt, cpp_name_data, &cpp_name_data::name,
    std::vector<type2tc>, cpp_name_data, &cpp_name_data::template_args>;

// Explicit instanciation for exprs.

template class esbmct::expr_methods<constant_int2t, constant_int_data,
    BigInt, constant_int_data, &constant_int_data::constant_value>;
template class esbmct::expr_methods<constant_fixedbv2t, constant_fixedbv_data,
    fixedbvt, constant_fixedbv_data, &constant_fixedbv_data::value>;
template class esbmct::expr_methods<constant_struct2t, constant_datatype_data,
    std::vector<expr2tc>, constant_datatype_data,
    &constant_datatype_data::datatype_members>;
template class esbmct::expr_methods<constant_union2t, constant_datatype_data,
    std::vector<expr2tc>, constant_datatype_data,
    &constant_datatype_data::datatype_members>;
template class esbmct::expr_methods<constant_bool2t, constant_bool_data,
    bool, constant_bool_data, &constant_bool_data::constant_value>;
template class esbmct::expr_methods<constant_array2t, constant_datatype_data,
    std::vector<expr2tc>, constant_datatype_data,
    &constant_datatype_data::datatype_members>;
template class esbmct::expr_methods<constant_array_of2t, constant_array_of_data,
    expr2tc, constant_array_of_data, &constant_array_of_data::initializer>;
template class esbmct::expr_methods<constant_string2t, constant_string_data,
    irep_idt, constant_string_data, &constant_string_data::value>;
template class esbmct::expr_methods<symbol2t, symbol_data,
    irep_idt, symbol_data, &symbol_data::thename,
    symbol_data::renaming_level, symbol_data, &symbol_data::rlevel,
    unsigned int, symbol_data, &symbol_data::level1_num,
    unsigned int, symbol_data, &symbol_data::level2_num,
    unsigned int, symbol_data, &symbol_data::thread_num,
    unsigned int, symbol_data, &symbol_data::node_num>;
template class esbmct::expr_methods<typecast2t, typecast_data,
    expr2tc, typecast_data, &typecast_data::from>;
template class esbmct::expr_methods<if2t, if_data,
    expr2tc, if_data, &if_data::cond,
    expr2tc, if_data, &if_data::true_value,
    expr2tc, if_data, &if_data::false_value>;
template class esbmct::expr_methods<equality2t, relation_data,
    expr2tc, relation_data, &relation_data::side_1,
    expr2tc, relation_data, &relation_data::side_2>;
template class esbmct::expr_methods<notequal2t, relation_data,
    expr2tc, relation_data, &relation_data::side_1,
    expr2tc, relation_data, &relation_data::side_2>;
template class esbmct::expr_methods<lessthan2t, relation_data,
    expr2tc, relation_data, &relation_data::side_1,
    expr2tc, relation_data, &relation_data::side_2>;
template class esbmct::expr_methods<greaterthan2t, relation_data,
    expr2tc, relation_data, &relation_data::side_1,
    expr2tc, relation_data, &relation_data::side_2>;
template class esbmct::expr_methods<lessthanequal2t, relation_data,
    expr2tc, relation_data, &relation_data::side_1,
    expr2tc, relation_data, &relation_data::side_2>;
template class esbmct::expr_methods<greaterthanequal2t, relation_data,
    expr2tc, relation_data, &relation_data::side_1,
    expr2tc, relation_data, &relation_data::side_2>;
template class esbmct::expr_methods<not2t, not_data,
    expr2tc, not_data, &not_data::value>;
template class esbmct::expr_methods<and2t, logic_2ops,
    expr2tc, logic_2ops, &logic_2ops::side_1,
    expr2tc, logic_2ops, &logic_2ops::side_2>;
template class esbmct::expr_methods<or2t, logic_2ops,
    expr2tc, logic_2ops, &logic_2ops::side_1,
    expr2tc, logic_2ops, &logic_2ops::side_2>;
template class esbmct::expr_methods<xor2t, logic_2ops,
    expr2tc, logic_2ops, &logic_2ops::side_1,
    expr2tc, logic_2ops, &logic_2ops::side_2>;
template class esbmct::expr_methods<implies2t, logic_2ops,
    expr2tc, logic_2ops, &logic_2ops::side_1,
    expr2tc, logic_2ops, &logic_2ops::side_2>;
template class esbmct::expr_methods<bitand2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<bitor2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<bitxor2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<bitnand2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<bitnor2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<bitnxor2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<lshr2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
template class esbmct::expr_methods<bitnot2t, bitnot_data,
    expr2tc, bitnot_data, &bitnot_data::value>;
template class esbmct::expr_methods<neg2t, arith_1op,
    expr2tc, arith_1op, &arith_1op::value>;
template class esbmct::expr_methods<abs2t, arith_1op,
    expr2tc, arith_1op, &arith_1op::value>;
template class esbmct::expr_methods<add2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<sub2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<mul2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<div2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<modulus2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<shl2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<ashr2t, arith_2ops,
    expr2tc, arith_2ops, &arith_2ops::side_1,
    expr2tc, arith_2ops, &arith_2ops::side_2>;
template class esbmct::expr_methods<same_object2t, same_object_data,
    expr2tc, same_object_data, &same_object_data::side_1,
    expr2tc, same_object_data, &same_object_data::side_2>;
template class esbmct::expr_methods<pointer_offset2t, pointer_ops,
    expr2tc, pointer_ops, &pointer_ops::ptr_obj>;
template class esbmct::expr_methods<pointer_object2t, pointer_ops,
    expr2tc, pointer_ops, &pointer_ops::ptr_obj>;
template class esbmct::expr_methods<address_of2t, pointer_ops,
    expr2tc, pointer_ops, &pointer_ops::ptr_obj>;
template class esbmct::expr_methods<byte_extract2t, byte_extract_data,
    expr2tc, byte_extract_data, &byte_extract_data::source_value,
    expr2tc, byte_extract_data, &byte_extract_data::source_offset,
    bool, byte_extract_data, &byte_extract_data::big_endian>;
template class esbmct::expr_methods<byte_update2t, byte_update_data,
    expr2tc, byte_update_data, &byte_update_data::source_value,
    expr2tc, byte_update_data, &byte_update_data::source_offset,
    expr2tc, byte_update_data, &byte_update_data::update_value,
    bool, byte_update_data, &byte_update_data::big_endian>;
template class esbmct::expr_methods<with2t, with_data,
    expr2tc, with_data, &with_data::source_value,
    expr2tc, with_data, &with_data::update_field,
    expr2tc, with_data, &with_data::update_value>;
template class esbmct::expr_methods<member2t, member_data,
    expr2tc, member_data, &member_data::source_value,
    irep_idt, member_data, &member_data::member>;
template class esbmct::expr_methods<index2t, index_data,
    expr2tc, index_data, &index_data::source_value,
    expr2tc, index_data, &index_data::index>;
template class esbmct::expr_methods<zero_string2t, string_ops,
    expr2tc, string_ops, &string_ops::string>;
template class esbmct::expr_methods<zero_length_string2t, string_ops,
    expr2tc, string_ops, &string_ops::string>;
template class esbmct::expr_methods<isnan2t, isnan_data,
    expr2tc, isnan_data, &isnan_data::value>;
template class esbmct::expr_methods<overflow2t, overflow_ops,
    expr2tc, overflow_ops, &overflow_ops::operand>;
template class esbmct::expr_methods<overflow_cast2t, overflow_cast_data,
    expr2tc, overflow_ops, &overflow_ops::operand,
    unsigned int, overflow_cast_data, &overflow_cast_data::bits>;
template class esbmct::expr_methods<overflow_neg2t, overflow_ops,
    expr2tc, overflow_ops, &overflow_ops::operand>;
template class esbmct::expr_methods<unknown2t, expr2t>;
template class esbmct::expr_methods<invalid2t, expr2t>;
template class esbmct::expr_methods<null_object2t, expr2t>;
template class esbmct::expr_methods<dynamic_object2t, dynamic_object_data,
    expr2tc, dynamic_object_data, &dynamic_object_data::instance,
    bool, dynamic_object_data, &dynamic_object_data::invalid,
    bool, dynamic_object_data, &dynamic_object_data::unknown>;
template class esbmct::expr_methods<dereference2t, dereference_data,
    expr2tc, dereference_data, &dereference_data::value>;
template class esbmct::expr_methods<valid_object2t, object_ops,
    expr2tc, object_ops, &object_ops::value>;
template class esbmct::expr_methods<deallocated_obj2t, object_ops,
    expr2tc, object_ops, &object_ops::value>;
template class esbmct::expr_methods<dynamic_size2t, object_ops,
    expr2tc, object_ops, &object_ops::value>;
template class esbmct::expr_methods<sideeffect2t, sideeffect_data,
    expr2tc, sideeffect_data, &sideeffect_data::operand,
    expr2tc, sideeffect_data, &sideeffect_data::size,
    std::vector<expr2tc>, sideeffect_data, &sideeffect_data::arguments,
    type2tc, sideeffect_data, &sideeffect_data::alloctype,
    sideeffect_data::allockind, sideeffect_data, &sideeffect_data::kind>;
template class esbmct::expr_methods<code_block2t, code_block_data,
    std::vector<expr2tc>, code_block_data, &code_block_data::operands>;
template class esbmct::expr_methods<code_assign2t, code_assign_data,
    expr2tc, code_assign_data, &code_assign_data::target,
    expr2tc, code_assign_data, &code_assign_data::source>;
template class esbmct::expr_methods<code_init2t, code_assign_data,
    expr2tc, code_assign_data, &code_assign_data::target,
    expr2tc, code_assign_data, &code_assign_data::source>;
template class esbmct::expr_methods<code_decl2t, code_decl_data,
    irep_idt, code_decl_data, &code_decl_data::value>;
template class esbmct::expr_methods<code_printf2t, code_printf_data,
    std::vector<expr2tc>, code_printf_data, &code_printf_data::operands>;
template class esbmct::expr_methods<code_expression2t, code_expression_data,
    expr2tc, code_expression_data, &code_expression_data::operand>;
template class esbmct::expr_methods<code_return2t, code_expression_data,
    expr2tc, code_expression_data, &code_expression_data::operand>;
template class esbmct::expr_methods<code_skip2t, expr2t>;
template class esbmct::expr_methods<code_free2t, code_expression_data,
    expr2tc, code_expression_data, &code_expression_data::operand>;
template class esbmct::expr_methods<code_goto2t, code_goto_data,
    irep_idt, code_goto_data, &code_goto_data::target>;
template class esbmct::expr_methods<object_descriptor2t, object_desc_data,
    expr2tc, object_desc_data, &object_desc_data::object,
    expr2tc, object_desc_data, &object_desc_data::offset,
    unsigned int, object_desc_data, &object_desc_data::alignment>;
template class esbmct::expr_methods<code_function_call2t, code_funccall_data,
    expr2tc, code_funccall_data, &code_funccall_data::ret,
    expr2tc, code_funccall_data, &code_funccall_data::function,
    std::vector<expr2tc>, code_funccall_data, &code_funccall_data::operands>;
template class esbmct::expr_methods<code_comma2t, code_comma_data,
    expr2tc, code_comma_data, &code_comma_data::side_1,
    expr2tc, code_comma_data, &code_comma_data::side_2>;
template class esbmct::expr_methods<invalid_pointer2t, pointer_ops,
    expr2tc, pointer_ops, &pointer_ops::ptr_obj>;
template class esbmct::expr_methods<buffer_size2t, buffer_size_data,
    expr2tc, buffer_size_data, &buffer_size_data::value>;
template class esbmct::expr_methods<code_asm2t, code_asm_data,
    irep_idt, code_asm_data, &code_asm_data::value>;
template class esbmct::expr_methods<code_cpp_del_array2t, code_expression_data,
    expr2tc, code_expression_data, &code_expression_data::operand>;
template class esbmct::expr_methods<code_cpp_delete2t, code_expression_data,
    expr2tc, code_expression_data, &code_expression_data::operand>;
template class esbmct::expr_methods<code_cpp_catch2t, code_cpp_catch_data,
    std::vector<irep_idt>, code_cpp_catch_data,
    &code_cpp_catch_data::exception_list>;
template class esbmct::expr_methods<code_cpp_throw2t, code_cpp_throw_data,
    expr2tc, code_cpp_throw_data, &code_cpp_throw_data::operand,
    std::vector<irep_idt>, code_cpp_throw_data,
    &code_cpp_throw_data::exception_list>;
template class esbmct::expr_methods<code_cpp_throw_decl2t,
         code_cpp_throw_decl_data,
    std::vector<irep_idt>, code_cpp_throw_decl_data,
    &code_cpp_throw_decl_data::exception_list>;
template class esbmct::expr_methods<code_cpp_throw_decl_end2t,
    code_cpp_throw_decl_data, std::vector<irep_idt>, code_cpp_throw_decl_data,
    &code_cpp_throw_decl_data::exception_list>;
template class esbmct::expr_methods<isinf2t,
    arith_1op, expr2tc, arith_1op, &arith_1op::value>;
template class esbmct::expr_methods<isnormal2t,
    arith_1op, expr2tc, arith_1op, &arith_1op::value>;
template class esbmct::expr_methods<concat2t, bit_2ops,
    expr2tc, bit_2ops, &bit_2ops::side_1,
    expr2tc, bit_2ops, &bit_2ops::side_2>;
