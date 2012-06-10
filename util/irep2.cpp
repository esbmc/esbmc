#include "irep2.h"
#include <stdarg.h>
#include <string.h>

#include "std_types.h"
#include "migrate.h"

#include <solvers/prop/prop_conv.h>

#include <boost/algorithm/string.hpp>
#include <boost/static_assert.hpp>

std::string
indent_str(unsigned int indent)
{
  return std::string(indent, ' ');
}

template <class T>
std::string
pretty_print_func(unsigned int indent, std::string ident, T obj)
{
  list_of_memberst memb = obj.tostring(indent);

  std::string indentstr = indent_str(indent);
  std::string exprstr = ident;

  for (list_of_memberst::const_iterator it = memb.begin(); it != memb.end();
       it++) {
    exprstr += "\n" + indentstr + it->first + " : " + it->second;
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
  "string"
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
  : type_id(id)
{
}

type2t::type2t(const type2t &ref)
  : type_id(ref.type_id)
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
  boost::crc_32_type crc;
  do_crc(crc);
  return crc.checksum();
}

void
type2t::do_crc(boost::crc_32_type &crc) const
{
  crc.process_byte(type_id);
  return;
}

unsigned int
bool_type2t::get_width(void) const
{
  return 1;
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

/*************************** Base expr2t definitions **************************/

expr2t::expr2t(const type2tc _type, expr_ids id)
  : expr_id(id), type(_type)
{
}

expr2t::expr2t(const expr2t &ref)
  : expr_id(ref.expr_id),
    type(ref.type)
{
}

void expr2t::convert_smt(prop_convt &obj, void *&arg) const
{ obj.convert_smt_expr(*this, arg); }


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
  expr_operands ops;

  list_operands(ops);

  for (expr_operands::const_iterator it = ops.begin(); it != ops.end(); it++) {
    unsigned long tmp = (***it).depth();
    num_nodes = std::max(num_nodes, tmp);
  }

  num_nodes++; // Count ourselves.
  return num_nodes;
}

unsigned long
expr2t::num_nodes(void) const
{
  unsigned long count = 0;
  expr_operands ops;

  list_operands(ops);

  for (expr_operands::const_iterator it = ops.begin(); it != ops.end(); it++)
    count += (***it).num_nodes();

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
  boost::crc_32_type crc;
  do_crc(crc);
  return crc.checksum();
}

void
expr2t::do_crc(boost::crc_32_type &crc) const
{
  crc.process_byte(expr_id);
  type->do_crc(crc);
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
  std::list<const expr2tc*> operands;
  std::list<expr2tc> newoperands;

  list_operands(operands);
  for (std::list<const expr2tc *>::iterator it = operands.begin();
       it != operands.end(); it++) {
    expr2tc tmp = (**it).get()->simplify();
    newoperands.push_back(tmp);
    if (!is_nil_expr(tmp))
      changed = true;
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
  Forall_operands2(it, expr_ops, new_us) {
    if ((*it2) == NULL)
      ; // No change in operand;
    else
      **it = *it2; // Operand changed; overwrite with new one.
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
  "from_bv_typecast",
  "to_bv_typecast"
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
  ret += std::string("\n") + indent_str(indent) + "type : "
         + type->pretty(indent + 2);
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

static const type2tc &
get_type_from_pool(const typet &val, std::map<const typet, type2tc> &map)
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

template <>
inline std::string
type_to_string<bool>(const bool &thebool, int indent __attribute__((unused)))
{
  return (thebool) ? "true" : "false";
}

template <>
inline std::string
type_to_string<unsigned int>(const unsigned int &theval,
                             int indent __attribute__((unused)))
{
  char buffer[64];
  snprintf(buffer, 63, "%d", theval);
  return std::string(buffer);
}

template <>
inline std::string
type_to_string<BigInt>(const BigInt &theint, int indent __attribute__((unused)))
{
  char buffer[256], *buf;

  buf = theint.as_string(buffer, 256);
  return std::string(buf);
}

template <>
inline std::string
type_to_string<fixedbvt>(const fixedbvt &theval,
                         int indent __attribute__((unused)))
{
  return theval.to_ansi_c_string();
}

template <>
inline std::string
type_to_string<std::vector<expr2tc> >(const std::vector<expr2tc> &theval,
                                     int indent)
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

template <>
inline std::string
type_to_string<std::vector<type2tc> >(const std::vector<type2tc> &theval,
                                      int indent)
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

template <>
inline std::string
type_to_string<std::vector<irep_idt> >(const std::vector<irep_idt> &theval,
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

template <>
inline std::string
type_to_string<expr2tc>(const expr2tc &theval, int indent)
{

  if (theval.get() != NULL)
   return theval->pretty(indent + 2);
  return "";
}

template <>
inline std::string
type_to_string<type2tc>(const type2tc &theval, int indent)
{

  if (theval.get() != NULL)
    return theval->pretty(indent + 2);
  else
    return "";
}

template <>
inline std::string
type_to_string<irep_idt>(const irep_idt &theval,
                         int indent __attribute__((unused)))
{
  return theval.as_string();
}

template <>
inline std::string
type_to_string<type2t::type_ids>(const type2t::type_ids &id,
                                 int indent __attribute__((unused)))
{
  return "";
}

template <>
inline std::string
type_to_string<const expr2t::expr_ids>(const expr2t::expr_ids &id,
                                 int indent __attribute__((unused)))
{
  return "";
}

template <>
inline bool
do_type_cmp<bool>(const bool &side1, const bool &side2)
{
  return (side1 == side2) ? true : false;
}

template <>
inline bool
do_type_cmp<unsigned int>(const unsigned int &side1, const unsigned int &side2)
{
  return (side1 == side2) ? true : false;
}

template <>
inline bool
do_type_cmp<BigInt>(const BigInt &side1, const BigInt &side2)
{
  // BigInt has its own equality operator.
  return (side1 == side2) ? true : false;
}

template <>
inline bool
do_type_cmp<fixedbvt>(const fixedbvt &side1, const fixedbvt &side2)
{
  return (side1 == side2) ? true : false;
}

template <>
inline bool
do_type_cmp<std::vector<expr2tc> >(const std::vector<expr2tc> &side1,
                                   const std::vector<expr2tc> &side2)
{
  return (side1 == side2);
}

template <>
inline bool
do_type_cmp<std::vector<type2tc> >(const std::vector<type2tc> &side1,
                                   const std::vector<type2tc> &side2)
{
  return (side1 == side2);
}

template <>
inline bool
do_type_cmp<std::vector<irep_idt> >(const std::vector<irep_idt> &side1,
                                    const std::vector<irep_idt> &side2)
{
  return (side1 == side2);
}

template <>
inline bool
do_type_cmp<expr2tc>(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // Catch null
  else if (side1.get() == NULL || side2.get() == NULL)
    return false;
  else
    return (side1 == side2);
}

template <>
inline bool
do_type_cmp<type2tc>(const type2tc &side1, const type2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // both null ptr check
  if (side1.get() == NULL || side2.get() == NULL)
    return false; // One of them is null, the other isn't
  return (side1 == side2);
}

template <>
inline bool
do_type_cmp<irep_idt>(const irep_idt &side1, const irep_idt &side2)
{
  return (side1 == side2);
}

template <>
inline bool
do_type_cmp<type2t::type_ids>(const type2t::type_ids &id,
                              const type2t::type_ids &id2)
{
  return true; // Dummy field comparison.
}

template <>
inline bool
do_type_cmp<const expr2t::expr_ids>(const expr2t::expr_ids &id,
                                    const expr2t::expr_ids &id2)
{
  return true; // Dummy field comparison.
}

template <>
inline int
do_type_lt<bool>(const bool &side1, const bool &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

template <>
inline int
do_type_lt<unsigned int>(const unsigned int &side1, const unsigned int &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

template <>
inline int
do_type_lt<BigInt>(const BigInt &side1, const BigInt &side2)
{
  // BigInt also has its own less than comparator.
  return side1.compare(side2);
}

template <>
inline int
do_type_lt<fixedbvt>(const fixedbvt &side1, const fixedbvt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side1 > side2)
    return 1;
  return 0;
}

template <>
inline int
do_type_lt<std::vector<expr2tc> >(const std::vector<expr2tc> &side1,
                                  const std::vector<expr2tc> &side2)
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

template <>
inline int
do_type_lt<std::vector<type2tc> >(const std::vector<type2tc> &side1,
                                  const std::vector<type2tc> &side2)
{

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

template <>
inline int
do_type_lt<std::vector<irep_idt> >(const std::vector<irep_idt> &side1,
                                  const std::vector<irep_idt> &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  return 0;
}

template <>
inline int
do_type_lt<expr2tc>(const expr2tc &side1, const expr2tc &side2)
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

template <>
inline int
do_type_lt<type2tc>(const type2tc &side1, const type2tc &side2)
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

template <>
inline int
do_type_lt<irep_idt>(const irep_idt &side1, const irep_idt &side2)
{
  if (side1 < side2)
    return -1;
  if (side2 < side1)
    return 1;
  return 0;
}

template <>
inline int
do_type_lt<type2t::type_ids>(const type2t::type_ids &id,
                             const type2t::type_ids &id2)
{
  return 0; // Dummy field comparison
}

template <>
inline int
do_type_lt<const expr2t::expr_ids>(const expr2t::expr_ids &id,
                                   const expr2t::expr_ids &id2)
{
  return 0; // Dummy field comparison
}

template <>
inline void
do_type_crc<bool>(const bool &thebool, boost::crc_32_type &crc)
{

  if (thebool)
    crc.process_byte(0);
  else
    crc.process_byte(1);
  return;
}

template <>
inline void
do_type_crc<unsigned int>(const unsigned int &theval, boost::crc_32_type &crc)
{

  crc.process_bytes(&theval, sizeof(theval));
  return;
}

template <>
inline void
do_type_crc<BigInt>(const BigInt &theint, boost::crc_32_type &crc)
{
  unsigned char buffer[256];

  if (theint.dump(buffer, sizeof(buffer))) {
    // Zero has no data in bigints.
    if (theint.is_zero())
      crc.process_byte(0);
    else
      crc.process_bytes(buffer, theint.get_len());
  } else {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return;
}

template <>
inline void
do_type_crc<fixedbvt>(const fixedbvt &theval, boost::crc_32_type &crc)
{

  do_type_crc<BigInt>(theval.to_integer(), crc);
  return;
}

template <>
inline void
do_type_crc<std::vector<expr2tc> >(const std::vector<expr2tc> &theval,
                                   boost::crc_32_type &crc)
{
  forall_exprs(it, theval)
    (*it)->do_crc(crc);
}

template <>
inline void
do_type_crc<std::vector<type2tc> >(const std::vector<type2tc> &theval,
                                   boost::crc_32_type &crc)
{
  forall_types(it, theval)
    (*it)->do_crc(crc);
}

template <>
inline void
do_type_crc<std::vector<irep_idt> >(const std::vector<irep_idt> &theval,
                                    boost::crc_32_type &crc)
{
  forall_names(it, theval)
    crc.process_bytes((*it).as_string().c_str(), (*it).as_string().size());
}

template <>
inline void
do_type_crc<expr2tc>(const expr2tc &theval, boost::crc_32_type &crc)
{

  if (theval.get() != NULL)
    theval->do_crc(crc);
  return;
}

template <>
inline void
do_type_crc<type2tc>(const type2tc &theval, boost::crc_32_type &crc)
{

  if (theval.get() != NULL)
    theval->do_crc(crc);
  return;
}

template <>
inline void
do_type_crc<irep_idt>(const irep_idt &theval, boost::crc_32_type &crc)
{

  crc.process_bytes(theval.as_string().c_str(), theval.as_string().size());
  return;
}

template <>
inline void
do_type_crc<type2t::type_ids>(const type2t::type_ids &i,boost::crc_32_type &crc)
{
  return; // Dummy field crc
}

template <>
inline void
do_type_crc<const expr2t::expr_ids>(const expr2t::expr_ids &i,
                                    boost::crc_32_type &crc)
{
  return; // Dummy field crc
}

template<> inline void do_type_list_operands<type2tc>(const type2tc &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<std::list<type2tc> >(const std::list<type2tc> &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<bool>(const bool &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<unsigned int>(const unsigned int &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<BigInt>(const BigInt &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<fixedbvt>(const fixedbvt &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<dstring>(const dstring &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<const expr2t::expr_ids>(const expr2t::expr_ids &theval __attribute__((unused)), std::list<const expr2tc*> &inp __attribute__((unused))) { return; }

template<> inline void do_type_list_operands<type2tc>(type2tc &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<std::list<type2tc> >(std::list<type2tc> &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<bool>(bool &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<unsigned int>(unsigned int &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<BigInt>(BigInt &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<fixedbvt>(fixedbvt &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<dstring>(dstring &theval __attribute__((unused)), std::list<expr2tc*> &inp __attribute__((unused))) { return; }
template<> inline void do_type_list_operands<const expr2t::expr_ids>(const expr2t::expr_ids &theval __attribute__((unused)), std::list< expr2tc*> &inp __attribute__((unused))) { return; }

template<>
inline void
do_type_list_operands<expr2tc>(expr2tc &theval,
                               std::list<expr2tc*> &inp)
{
  if (is_nil_expr(theval))
    return;

  inp.push_back(&theval);
}

template<>
inline void
do_type_list_operands<std::vector<expr2tc> >(std::vector<expr2tc> &theval,
                      std::list<expr2tc*> &inp)
{
  for (std::vector<expr2tc>::iterator it = theval.begin(); it != theval.end();
       it++) {
    if (!is_nil_expr(*it))
      inp.push_back(&(*it));
  }
}

template<>
inline void
do_type_list_operands<expr2tc>(const expr2tc &theval,
                               std::list<const expr2tc *> &inp)
{
  if (is_nil_expr(theval))
    return;

  inp.push_back(&theval);
}

template<>
inline void
do_type_list_operands<std::vector<expr2tc> >(const std::vector<expr2tc> &theval,
                      std::list<const expr2tc *> &inp)
{
  for (std::vector<expr2tc>::const_iterator it = theval.begin();
       it != theval.end(); it++) {
    if (!is_nil_expr(*it))
      inp.push_back(&(*it));
  }
}

template <class derived,class field1, class field2, class field3, class field4>
void
esbmct::expr<derived, field1, field2, field3, field4>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived,class field1, class field2, class field3, class field4>
expr2tc
esbmct::expr<derived, field1, field2, field3, field4>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived,class field1, class field2, class field3, class field4>
expr2t *
esbmct::expr<derived, field1, field2, field3, field4>::clone_raw(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return new_obj;
}

template <class derived,class field1, class field2, class field3, class field4>
list_of_memberst
esbmct::expr<derived, field1, field2, field3, field4>::tostring(unsigned int indent) const
{
  list_of_memberst thevector;
  field1::fieldtype::tostring(thevector, indent);
  field2::fieldtype::tostring(thevector, indent);
  field3::fieldtype::tostring(thevector, indent);
  field4::fieldtype::tostring(thevector, indent);
  return thevector;
}

template <class derived,class field1, class field2, class field3, class field4>
bool
esbmct::expr<derived, field1, field2, field3, field4>::cmp(const expr2t &ref)const
{
  const derived &ref2 = static_cast<const derived &>(ref);

  if (!field1::fieldtype::cmp(
        static_cast<const typename field1::fieldtype &>(ref2)))
    return false;

  if (!field2::fieldtype::cmp(
        static_cast<const typename field2::fieldtype &>(ref2)))
    return false;

  if (!field3::fieldtype::cmp(
        static_cast<const typename field3::fieldtype &>(ref2)))
    return false;

  if (!field4::fieldtype::cmp(
        static_cast<const typename field4::fieldtype &>(ref2)))
    return false;

  return true;
}

template <class derived,class field1, class field2, class field3, class field4>
int
esbmct::expr<derived, field1, field2, field3, field4>::lt(const expr2t &ref)const
{
  int tmp;
  const derived &ref2 = static_cast<const derived &>(ref);

  tmp = field1::fieldtype::lt(
                static_cast<const typename field1::fieldtype &>(ref2));
  if (tmp != 0)
    return tmp;

  tmp = field2::fieldtype::lt(
                static_cast<const typename field2::fieldtype &>(ref2));
  if (tmp != 0)
    return tmp;

  tmp = field3::fieldtype::lt(
                static_cast<const typename field3::fieldtype &>(ref2));
  if (tmp != 0)
    return tmp;

  tmp = field4::fieldtype::lt(
                static_cast<const typename field4::fieldtype &>(ref2));

  return tmp;
}

template <class derived,class field1, class field2, class field3, class field4>
void
esbmct::expr<derived, field1, field2, field3, field4>::do_crc
          (boost::crc_32_type &crc) const
{

  expr2t::do_crc(crc);
  field1::fieldtype::do_crc(crc);
  field2::fieldtype::do_crc(crc);
  field3::fieldtype::do_crc(crc);
  field4::fieldtype::do_crc(crc);
  return;
}

template <class derived,class field1, class field2, class field3, class field4>
void
esbmct::expr<derived, field1, field2, field3, field4>::list_operands
          (std::list<const expr2tc *> &inp) const
{

  field1::fieldtype::list_operands(inp);
  field2::fieldtype::list_operands(inp);
  field3::fieldtype::list_operands(inp);
  field4::fieldtype::list_operands(inp);
  return;
}

template <class derived,class field1, class field2, class field3, class field4>
void
esbmct::expr<derived, field1, field2, field3, field4>::list_operands
          (std::list<expr2tc*> &inp)
{

  field1::fieldtype::list_operands(inp);
  field2::fieldtype::list_operands(inp);
  field3::fieldtype::list_operands(inp);
  field4::fieldtype::list_operands(inp);
  return;
}

template <typename T>
void
do_type2string(const T &thething, unsigned int idx,
               std::string (&names)[esbmct::num_type_fields],
               list_of_memberst &vec, unsigned int indent)
{
  vec.push_back(member_entryt(names[idx], type_to_string<T>(thething, indent)));
}

template <>
void
do_type2string<type2t::type_ids>(const type2t::type_ids &thething,
                                 unsigned int idx,
                                 std::string (&names)[esbmct::num_type_fields],
                                 list_of_memberst &vec, unsigned int indent)
{
  // Do nothing; this is a dummy member.
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
void
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
expr2tc
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
expr2t *
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::clone_raw(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return new_obj;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
list_of_memberst
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::tostring(unsigned int indent) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  list_of_memberst thevector;
  do_type2string<field1_type>(derived_this->*field1_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field2_type>(derived_this->*field2_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field3_type>(derived_this->*field3_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field4_type>(derived_this->*field4_ptr, 0,
                              derived_this->field_names, thevector, indent);
  return thevector;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
bool
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::cmp(const expr2t &ref)const
{
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  if (!do_type_cmp<field1_type>(derived_this->*field1_ptr, ref2->*field1_ptr))
    return false;

  if (!do_type_cmp<field2_type>(derived_this->*field2_ptr, ref2->*field2_ptr))
    return false;

  if (!do_type_cmp<field3_type>(derived_this->*field3_ptr, ref2->*field3_ptr))
    return false;

  if (!do_type_cmp<field4_type>(derived_this->*field4_ptr, ref2->*field4_ptr))
    return false;

  return true;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
int
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::lt(const expr2t &ref)const
{
  int tmp;
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  tmp = do_type_lt<field1_type>(derived_this->*field1_ptr, ref2->*field1_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt<field2_type>(derived_this->*field2_ptr, ref2->*field2_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt<field3_type>(derived_this->*field3_ptr, ref2->*field3_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt<field4_type>(derived_this->*field4_ptr, ref2->*field4_ptr);
  if (tmp != 0)
    return tmp;

  return tmp;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
void
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::do_crc
          (boost::crc_32_type &crc) const
{
  const derived *derived_this = static_cast<const derived*>(this);

  derived_this->expr2t::do_crc(crc);
  do_type_crc<field1_type>(derived_this->*field1_ptr, crc);
  do_type_crc<field2_type>(derived_this->*field2_ptr, crc);
  do_type_crc<field3_type>(derived_this->*field3_ptr, crc);
  do_type_crc<field4_type>(derived_this->*field4_ptr, crc);
  return;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
void
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::list_operands
          (std::list<const expr2tc *> &inp) const
{
  const derived *derived_this = static_cast<const derived*>(this);

  do_type_list_operands<field1_type>(derived_this->*field1_ptr, inp);
  do_type_list_operands<field2_type>(derived_this->*field2_ptr, inp);
  do_type_list_operands<field3_type>(derived_this->*field3_ptr, inp);
  do_type_list_operands<field4_type>(derived_this->*field4_ptr, inp);
  return;
}

template <class derived, class subclass,
        typename field1_type, class field1_class,
        field1_type field1_class::*field1_ptr,
        typename field2_type, class field2_class,
        field2_type field2_class::*field2_ptr,
        typename field3_type, class field3_class,
        field3_type field3_class::*field3_ptr,
        typename field4_type, class field4_class,
        field4_type field4_class::*field4_ptr>
void
esbmct::expr_methods<derived, subclass, field1_type, field1_class, field1_ptr, field2_type, field2_class, field2_ptr, field3_type, field3_class, field3_ptr, field4_type, field4_class, field4_ptr>::list_operands
          (std::list<expr2tc*> &inp)
{
  derived *derived_this = static_cast<derived*>(this);

  do_type_list_operands<field1_type>(derived_this->*field1_ptr, inp);
  do_type_list_operands<field2_type>(derived_this->*field2_ptr, inp);
  do_type_list_operands<field3_type>(derived_this->*field3_ptr, inp);
  do_type_list_operands<field4_type>(derived_this->*field4_ptr, inp);
  return;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr>
void
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr>
      ::convert_smt_type(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_type(*new_this, arg);
  return;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr>
type2tc
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr>
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
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr>
list_of_memberst
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr>
      ::tostring(unsigned int indent) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  list_of_memberst thevector;
  do_type2string<field1_type>(derived_this->*field1_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field2_type>(derived_this->*field2_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field3_type>(derived_this->*field3_ptr, 0,
                              derived_this->field_names, thevector, indent);
  do_type2string<field4_type>(derived_this->*field4_ptr, 0,
                              derived_this->field_names, thevector, indent);
  return thevector;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr>
bool
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr>
      ::cmp(const type2t &ref) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  if (!do_type_cmp<field1_type>(derived_this->*field1_ptr, ref2->*field1_ptr))
    return false;

  if (!do_type_cmp<field2_type>(derived_this->*field2_ptr, ref2->*field2_ptr))
    return false;

  if (!do_type_cmp<field3_type>(derived_this->*field3_ptr, ref2->*field3_ptr))
    return false;

  if (!do_type_cmp<field4_type>(derived_this->*field4_ptr, ref2->*field4_ptr))
    return false;

  return true;
}


template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr>
int
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr>
      ::lt(const type2t &ref)const
{
  int tmp;
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);

  tmp = do_type_lt<field1_type>(derived_this->*field1_ptr, ref2->*field1_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt<field2_type>(derived_this->*field2_ptr, ref2->*field2_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt<field3_type>(derived_this->*field3_ptr, ref2->*field3_ptr);
  if (tmp != 0)
    return tmp;

  tmp = do_type_lt<field4_type>(derived_this->*field4_ptr, ref2->*field4_ptr);
  if (tmp != 0)
    return tmp;

  return tmp;
}

template <class derived, class subclass,
  class field1_type, class field1_class, field1_type field1_class::*field1_ptr,
  class field2_type, class field2_class, field2_type field2_class::*field2_ptr,
  class field3_type, class field3_class, field3_type field3_class::*field3_ptr,
  class field4_type, class field4_class, field4_type field4_class::*field4_ptr>
void
esbmct::type_methods<derived, subclass, field1_type, field1_class, field1_ptr,
                                        field2_type, field2_class, field2_ptr,
                                        field3_type, field3_class, field3_ptr,
                                        field4_type, field4_class, field4_ptr>
      ::do_crc (boost::crc_32_type &crc) const
{

  const derived *derived_this = static_cast<const derived*>(this);

  derived_this->type2t::do_crc(crc);
  do_type_crc<field1_type>(derived_this->*field1_ptr, crc);
  do_type_crc<field2_type>(derived_this->*field2_ptr, crc);
  do_type_crc<field3_type>(derived_this->*field3_ptr, crc);
  do_type_crc<field4_type>(derived_this->*field4_ptr, crc);
  return;
}

std::string bool_type2t::field_names [esbmct::num_type_fields]  = {"","","",""};
std::string empty_type2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", ""};
std::string symbol_type2t::field_names [esbmct::num_type_fields]  =
{ "symbol_name", "", "", ""};
std::string struct_type2t::field_names [esbmct::num_type_fields]  =
{ "members", "member_names", "typename", ""};
std::string union_type2t::field_names [esbmct::num_type_fields]  =
{ "members", "member_names", "typename", ""};
std::string unsignedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", ""};
std::string signedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", ""};
std::string code_type2t::field_names [esbmct::num_type_fields]  =
{ "arguments", "ret_type", "argument_names", "ellipsis"};
std::string array_type2t::field_names [esbmct::num_type_fields]  =
{ "subtype", "array_size", "size_is_infinite", ""};
std::string pointer_type2t::field_names [esbmct::num_type_fields]  =
{ "subtype", "", "", ""};
std::string fixedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "integer_bits", "", ""};
std::string string_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", ""};


// Exprs

std::string constant_int2t::field_names [esbmct::num_type_fields]  =
{ "constant_value", "", "", ""};
std::string constant_fixedbv2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", ""};
std::string constant_struct2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", ""};
std::string constant_union2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", ""};
std::string constant_bool2t::field_names [esbmct::num_type_fields]  =
{ "constant_value", "", "", ""};
std::string constant_array2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", ""};
std::string constant_array_of2t::field_names [esbmct::num_type_fields]  =
{ "initializer", "", "", ""};
std::string constant_string2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", ""};
std::string symbol2t::field_names [esbmct::num_type_fields]  =
{ "name", "", "", ""};
std::string typecast2t::field_names [esbmct::num_type_fields]  =
{ "from", "", "", ""};
std::string to_bv_typecast2t::field_names [esbmct::num_type_fields]  =
{ "from", "", "", ""};
std::string from_bv_typecast2t::field_names [esbmct::num_type_fields]  =
{ "from", "", "", ""};
std::string if2t::field_names [esbmct::num_type_fields]  =
{ "cond", "true_value", "false_value", ""};

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
    irep_idt, symbol_data, &symbol_data::name>;
template class esbmct::expr_methods<typecast2t, typecast_data,
    expr2tc, typecast_data, &typecast_data::from>;
template class esbmct::expr_methods<to_bv_typecast2t, typecast_data,
    expr2tc, typecast_data, &typecast_data::from>;
template class esbmct::expr_methods<from_bv_typecast2t, typecast_data,
    expr2tc, typecast_data, &typecast_data::from>;
template class esbmct::expr_methods<if2t, if_data,
    expr2tc, if_data, &if_data::cond,
    expr2tc, if_data, &if_data::true_value,
    expr2tc, if_data, &if_data::false_value>;
