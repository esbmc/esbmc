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
  std::cerr <<"Fetching width of symbol type - invalid operation" << std::endl;
  abort();
}

unsigned int
cpp_name_type2t::get_width(void) const
{
  std::cerr << "Fetching width of cpp_name type - invalid operation" << std::endl;
  abort();
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
  // taking the address of some /completely/ arbitrary pice of data, by
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
  new_us.get()->Foreach_operand([this, &it2] (expr2tc &e) {
      if ((*it2) == NULL)
        ; // No change in operand;
      else
        e = *it2; // Operand changed; overwrite with new one.
      it2++;
    }
  );

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
  ret += std::string("\n") + indent_str(indent) + "* type : "
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

// XXX investigate performance implications of this cache
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
         (data == sideeffect_data::allockind::alloca) ? "alloca" :
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
do_get_sub_expr_nc(T &item __attribute__((unused)),
                unsigned int idx __attribute__((unused)),
                unsigned int &it __attribute__((unused)),
                expr2tc *&ptr __attribute__((unused)))
{
  return false;
}

template <>
bool
do_get_sub_expr_nc<expr2tc>(expr2tc &item, unsigned int idx, unsigned int &it,
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
do_get_sub_expr_nc<std::vector<expr2tc>>(std::vector<expr2tc> &item,
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

typedef std::size_t lolnoop;
inline std::size_t
hash_value(lolnoop val)
{
  return val;
}

// Local template for implementing delegate calling, with type dependency.
template <typename T, typename U>
void
call_expr_delegate(T &ref, U &f)
{
  // Don't do anything normally.
  (void)ref;
  (void)f;
  return;
}

template <>
void
call_expr_delegate<const expr2tc,expr2t::const_op_delegate>
                  (const expr2tc &ref, expr2t::const_op_delegate &f)
{
  f(ref);
  return;
}

template <>
void
call_expr_delegate<expr2tc, expr2t::op_delegate>
                  (expr2tc &ref, expr2t::op_delegate &f)
{
  f(ref);
  return;
}

template <>
void
call_expr_delegate<const std::vector<expr2tc>, expr2t::const_op_delegate>
                 (const std::vector<expr2tc> &ref, expr2t::const_op_delegate &f)
{
  for (const expr2tc &r : ref)
    f(r);

  return;
}

template <>
void
call_expr_delegate<std::vector<expr2tc>, expr2t::op_delegate>
                  (std::vector<expr2tc> &ref, expr2t::op_delegate &f)
{
  for (expr2tc &r : ref)
    f(r);

  return;
}

/************************ Second attempt at irep templates ********************/

// Implementations of common methods, recursively.

// Top level type method definition (above recursive def)
// exprs

template <class derived, class baseclass, typename traits, typename enable>
const expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, enable>::get_sub_expr(unsigned int i) const
{
  return superclass::get_sub_expr_rec(0, i); // Skips expr_id
}

template <class derived, class baseclass, typename traits, typename enable>
expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, enable>::get_sub_expr_nc(unsigned int i)
{
  return superclass::get_sub_expr_nc_rec(0, i); // Skips expr_id
}

template <class derived, class baseclass, typename traits, typename enable>
unsigned int
esbmct::expr_methods2<derived, baseclass, traits, enable>::get_num_sub_exprs(void) const
{
  return superclass::get_num_sub_exprs_rec(); // Skips expr_id
}

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::expr_methods2<derived, baseclass, traits, enable>::foreach_operand_impl_const(expr2t::const_op_delegate &f) const
{
  superclass::foreach_operand_impl_const_rec(f);
}

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::expr_methods2<derived, baseclass, traits, enable>::foreach_operand_impl(expr2t::op_delegate &f)
{
  superclass::foreach_operand_impl_rec(f);
}

// Types

template <class derived, class baseclass, typename traits, typename enable>
auto
esbmct::irep_methods2<derived, baseclass, traits, enable>::clone(void) const -> container2tc
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return container2tc(new_obj);
}

template <class derived, class baseclass, typename traits, typename enable>
list_of_memberst
esbmct::irep_methods2<derived, baseclass, traits, enable>::tostring(unsigned int indent) const
{
  list_of_memberst thevector;

  superclass::tostring_rec(0, thevector, indent); // Skips type_id / expr_id
  return thevector;
}

template <class derived, class baseclass, typename traits, typename enable>
bool
esbmct::irep_methods2<derived, baseclass, traits, enable>::cmp(const base2t &ref) const
{
  return cmp_rec(ref); // _includes_ type_id / expr_id
}

template <class derived, class baseclass, typename traits, typename enable>
int
esbmct::irep_methods2<derived, baseclass, traits, enable>::lt(const base2t &ref) const
{
  return lt_rec(ref); // _includes_ type_id / expr_id
}

template <class derived, class baseclass, typename traits,  typename enable>
size_t
esbmct::irep_methods2<derived, baseclass, traits,  enable>::do_crc(size_t seed) const
{

  if (this->crc_val != 0) {
    boost::hash_combine(seed, (lolnoop)this->crc_val);
    return seed;
  }

  // Starting from 0, pass a crc value through all the sub-fields of this
  // expression. Store it into crc_val. Don't allow the input seed to affect
  // this calculation, as the crc value needs to uniquely identify _this_
  // expression.
  assert(this->crc_val == 0);

  do_crc_rec(); // _includes_ type_id / expr_id

  // Finally, combine the crc of this expr with the input seed, and return
  boost::hash_combine(seed, (lolnoop)this->crc_val);
  return seed;
}

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::irep_methods2<derived, baseclass, traits, enable>::hash(crypto_hash &hash) const
{

  hash_rec(hash); // _includes_ type_id / expr_id
  return;
}

// The, *actual* recursive defs

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::irep_methods2<derived, baseclass, traits, enable>::tostring_rec(unsigned int idx, list_of_memberst &vec, unsigned int indent) const
{
  // Skip over type fields in expressions. Alas, this is a design oversight,
  // without this we would screw up the field name list.
  // It escapes me why this isn't printed here anyway, it gets printed in the
  // end.
  if (std::is_same<cur_type, type2tc>::value && std::is_base_of<expr2t,derived>::value) {
    superclass::tostring_rec(idx, vec, indent);
    return;
  }

  // Insert our particular member to string list.
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;
  do_type2string<cur_type>(derived_this->*m_ptr, idx, derived_this->field_names, vec, indent);

  // Recurse
  superclass::tostring_rec(idx + 1, vec, indent);
}

template <class derived, class baseclass, typename traits, typename enable>
bool
esbmct::irep_methods2<derived, baseclass, traits, enable>::cmp_rec(const base2t &ref) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  if (!do_type_cmp(derived_this->*m_ptr, ref2->*m_ptr))
    return false;

  return superclass::cmp_rec(ref);
}

template <class derived, class baseclass, typename traits, typename enable>
int
esbmct::irep_methods2<derived, baseclass, traits, enable>::lt_rec(const base2t &ref) const
{
  int tmp;
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  tmp = do_type_lt(derived_this->*m_ptr, ref2->*m_ptr);
  if (tmp != 0)
    return tmp;

  return superclass::lt_rec(ref);
}

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::irep_methods2<derived, baseclass, traits, enable>::do_crc_rec() const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  size_t tmp = do_type_crc(derived_this->*m_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);

  superclass::do_crc_rec();
}

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::irep_methods2<derived, baseclass, traits, enable>::hash_rec(crypto_hash &hash) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;
  do_type_hash(derived_this->*m_ptr, hash);

  superclass::hash_rec(hash);
}

template <class derived, class baseclass, typename traits, typename enable>
const expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, enable>::get_sub_expr_rec(unsigned int cur_idx, unsigned int desired) const
{
  const expr2tc *ptr;
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if (do_get_sub_expr(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_rec(cur_idx, desired);
}

template <class derived, class baseclass, typename traits, typename enable>
expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, enable>::get_sub_expr_nc_rec(unsigned int cur_idx, unsigned int desired)
{
  expr2tc *ptr;
  derived *derived_this = static_cast<derived*>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if (do_get_sub_expr_nc(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_nc_rec(cur_idx, desired);
}

template <class derived, class baseclass, typename traits, typename enable>
unsigned int
esbmct::irep_methods2<derived, baseclass, traits, enable>::get_num_sub_exprs_rec(void) const
{
  unsigned int num = 0;
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  num = do_count_sub_exprs(derived_this->*m_ptr);
  return num + superclass::get_num_sub_exprs_rec();
}

// Operand iteration specialized for expr2tc: call delegate.
template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::irep_methods2<derived, baseclass, traits, enable>::foreach_operand_impl_const_rec(expr2t::const_op_delegate &f) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_const_rec(f);
}

template <class derived, class baseclass, typename traits, typename enable>
void
esbmct::irep_methods2<derived, baseclass, traits, enable>::foreach_operand_impl_rec(expr2t::op_delegate &f)
{
  derived *derived_this = static_cast<derived*>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_rec(f);
}

/********************** Constants and explicit instantiations *****************/

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

template class esbmct::irep_methods2<bool_type2t, type2t, typename esbmct::type2t_default_traits::type>;
template class esbmct::irep_methods2<empty_type2t, type2t, typename esbmct::type2t_default_traits::type>;
template class esbmct::irep_methods2<symbol_type2t, symbol_type_data, symbol_type_data::traits::type>;
template class esbmct::irep_methods2<struct_type2t, struct_union_data, struct_union_data::traits::type>;
template class esbmct::irep_methods2<union_type2t, struct_union_data, struct_union_data::traits::type>;
template class esbmct::irep_methods2<unsignedbv_type2t, bv_data, bv_data::traits::type>;
template class esbmct::irep_methods2<signedbv_type2t, bv_data, bv_data::traits::type>;
template class esbmct::irep_methods2<code_type2t, code_data, code_data::traits::type>;
template class esbmct::irep_methods2<array_type2t, array_data, array_data::traits::type>;
template class esbmct::irep_methods2<pointer_type2t, pointer_data, pointer_data::traits::type>;
template class esbmct::irep_methods2<fixedbv_type2t, fixedbv_data, fixedbv_data::traits::type>;
template class esbmct::irep_methods2<string_type2t, string_data, string_data::traits::type>;
template class esbmct::irep_methods2<cpp_name_type2t, cpp_name_data, cpp_name_data::traits::type>;

// Explicit instanciation for exprs.

// XXX workaround: borrow a macro from irep2.h to avoid retyping all of this.
// Use for explicit instantiation.

#undef irep_typedefs
#undef irep_typedefs_empty

#define irep_typedefs(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits::type>;

#define irep_typedefs_empty(basename, superclass) \
  template class  esbmct::expr_methods2<basename##2t, superclass, esbmct::expr2t_default_traits::type>;

irep_typedefs(constant_int, constant_int_data);
irep_typedefs(constant_fixedbv, constant_fixedbv_data);
irep_typedefs(constant_struct, constant_datatype_data);
irep_typedefs(constant_union, constant_datatype_data);
irep_typedefs(constant_array, constant_datatype_data);
irep_typedefs(constant_bool, constant_bool_data);
irep_typedefs(constant_array_of, constant_array_of_data);
irep_typedefs(constant_string, constant_string_data);
irep_typedefs(symbol, symbol_data);
irep_typedefs(typecast,typecast_data);
irep_typedefs(if, if_data);
irep_typedefs(equality, relation_data);
irep_typedefs(notequal, relation_data);
irep_typedefs(lessthan, relation_data);
irep_typedefs(greaterthan, relation_data);
irep_typedefs(lessthanequal, relation_data);
irep_typedefs(greaterthanequal, relation_data);
irep_typedefs(not, not_data);
irep_typedefs(and, logic_2ops);
irep_typedefs(or, logic_2ops);
irep_typedefs(xor, logic_2ops);
irep_typedefs(implies, logic_2ops);
irep_typedefs(bitand, bit_2ops);
irep_typedefs(bitor, bit_2ops);
irep_typedefs(bitxor, bit_2ops);
irep_typedefs(bitnand, bit_2ops);
irep_typedefs(bitnor, bit_2ops);
irep_typedefs(bitnxor, bit_2ops);
irep_typedefs(lshr, bit_2ops);
irep_typedefs(bitnot, bitnot_data);
irep_typedefs(neg, arith_1op);
irep_typedefs(abs, arith_1op);
irep_typedefs(add, arith_2ops);
irep_typedefs(sub, arith_2ops);
irep_typedefs(mul, arith_2ops);
irep_typedefs(div, arith_2ops);
irep_typedefs(modulus, arith_2ops);
irep_typedefs(shl, arith_2ops);
irep_typedefs(ashr, arith_2ops);
irep_typedefs(same_object, same_object_data);
irep_typedefs(pointer_offset, pointer_ops);
irep_typedefs(pointer_object, pointer_ops);
irep_typedefs(address_of, pointer_ops);
irep_typedefs(byte_extract, byte_extract_data);
irep_typedefs(byte_update, byte_update_data);
irep_typedefs(with, with_data);
irep_typedefs(member, member_data);
irep_typedefs(index, index_data);
irep_typedefs(isnan, isnan_data);
irep_typedefs(overflow, overflow_ops);
irep_typedefs(overflow_cast, overflow_cast_data);
irep_typedefs(overflow_neg, overflow_ops);
irep_typedefs_empty(unknown, expr2t);
irep_typedefs_empty(invalid, expr2t);
irep_typedefs_empty(null_object, expr2t);
irep_typedefs(dynamic_object, dynamic_object_data);
irep_typedefs(dereference, dereference_data);
irep_typedefs(valid_object, object_ops);
irep_typedefs(deallocated_obj, object_ops);
irep_typedefs(dynamic_size, object_ops);
irep_typedefs(sideeffect, sideeffect_data);
irep_typedefs(code_block, code_block_data);
irep_typedefs(code_assign, code_assign_data);
irep_typedefs(code_init, code_assign_data);
irep_typedefs(code_decl, code_decl_data);
irep_typedefs(code_printf, code_printf_data);
irep_typedefs(code_expression, code_expression_data);
irep_typedefs(code_return, code_expression_data);
irep_typedefs_empty(code_skip, expr2t);
irep_typedefs(code_free, code_expression_data);
irep_typedefs(code_goto, code_goto_data);
irep_typedefs(object_descriptor, object_desc_data);
irep_typedefs(code_function_call, code_funccall_data);
irep_typedefs(code_comma, code_comma_data);
irep_typedefs(invalid_pointer, invalid_pointer_ops);
irep_typedefs(code_asm, code_asm_data);
irep_typedefs(code_cpp_del_array, code_expression_data);
irep_typedefs(code_cpp_delete, code_expression_data);
irep_typedefs(code_cpp_catch, code_cpp_catch_data);
irep_typedefs(code_cpp_throw, code_cpp_throw_data);
irep_typedefs(code_cpp_throw_decl, code_cpp_throw_decl_data);
irep_typedefs(code_cpp_throw_decl_end, code_cpp_throw_decl_data);
irep_typedefs(isinf, isinf_data);
irep_typedefs(isnormal, isinf_data);
irep_typedefs(concat, bit_2ops);
