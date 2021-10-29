#include <memory>
//#include <ac_config.h>
#include <boost/functional/hash.hpp>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/std_types.h>
#include <util/message/format.h>
#include <util/message/default_message.h>

template <typename T>
class register_irep_methods;

std::string indent_str(unsigned int indent)
{
  return std::string(indent, ' ');
}

template <class T>
std::string pretty_print_func(unsigned int indent, std::string ident, T obj)
{
  list_of_memberst memb = obj.tostring(indent + 2);

  std::string indentstr = indent_str(indent);
  std::string exprstr = std::move(ident);

  for(list_of_memberst::const_iterator it = memb.begin(); it != memb.end();
      it++)
  {
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
  "floatbv",
  "string",
  "cpp_name"};
// If this fires, you've added/removed a type id, and need to update the list
// above (which is ordered according to the enum list)
static_assert(
  sizeof(type_names) == (type2t::end_type_id * sizeof(char *)),
  "Missing type name");

std::string get_type_id(const type2t &type)
{
  return std::string(type_names[type.type_id]);
}

type2t::type2t(type_ids id)
  : std::enable_shared_from_this<type2t>(), type_id(id), crc_val(0)
{
}

bool type2t::operator==(const type2t &ref) const
{
  return cmpchecked(ref);
}

bool type2t::operator!=(const type2t &ref) const
{
  return !cmpchecked(ref);
}

bool type2t::operator<(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if(tmp < 0)
    return true;
  else if(tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

int type2t::ltchecked(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if(tmp != 0)
    return tmp;

  return lt(ref);
}

bool type2t::cmpchecked(const type2t &ref) const
{
  if(type_id == ref.type_id)
    return cmp(ref);

  return false;
}

int type2t::lt(const type2t &ref) const
{
  if(type_id < ref.type_id)
    return -1;
  if(type_id > ref.type_id)
    return 1;
  return 0;
}

std::string type2t::pretty(unsigned int indent) const
{
  return pretty_print_func<const type2t &>(indent, type_names[type_id], *this);
}

void type2t::dump() const
{
  default_message msg;
  msg.debug(pretty(0));
}

size_t type2t::crc() const
{
  return do_crc();
}

size_t type2t::do_crc() const
{
  boost::hash_combine(this->crc_val, (uint8_t)type_id);
  return this->crc_val;
}

void type2t::hash(crypto_hash &hash) const
{
  static_assert(type2t::end_type_id < 256, "Type id overflow");
  uint8_t tid = type_id;
  hash.ingest(&tid, sizeof(tid));
}

unsigned int bool_type2t::get_width() const
{
  // For the purpose of the byte representating memory model
  return 8;
}

unsigned int bv_data::get_width() const
{
  return width;
}

unsigned int array_type2t::get_width() const
{
  // Two edge cases: the array can have infinite size, or it can have a dynamic
  // size that's determined by the solver.
  if(size_is_infinite)
    throw new inf_sized_array_excp();

  if(array_size->expr_id != expr2t::constant_int_id)
    throw new dyn_sized_array_excp(array_size);

  // Otherwise, we can multiply the size of the subtype by the number of elements.
  unsigned int sub_width = subtype->get_width();

  const expr2t *elem_size = array_size.get();
  const constant_int2t *const_elem_size =
    dynamic_cast<const constant_int2t *>(elem_size);
  assert(const_elem_size != nullptr);
  unsigned long num_elems = const_elem_size->as_ulong();

  return num_elems * sub_width;
}

unsigned int pointer_type2t::get_width() const
{
  return config.ansi_c.pointer_width;
}

unsigned int empty_type2t::get_width() const
{
  throw new symbolic_type_excp();
}

unsigned int symbol_type2t::get_width() const
{
  assert(0 && "Fetching width of symbol type - invalid operation");
  abort();
}

unsigned int cpp_name_type2t::get_width() const
{
  assert(0 && "Fetching width of cpp_name type - invalid operation");
  abort();
}

unsigned int struct_type2t::get_width() const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for(it = members.begin(); it != members.end(); it++)
    width += (*it)->get_width();

  return width;
}

unsigned int union_type2t::get_width() const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for(it = members.begin(); it != members.end(); it++)
    width = std::max(width, (*it)->get_width());

  return width;
}

unsigned int fixedbv_type2t::get_width() const
{
  return width;
}

unsigned int floatbv_type2t::get_width() const
{
  return fraction + exponent + 1;
}

unsigned int code_data::get_width() const
{
  throw new symbolic_type_excp();
}

unsigned int string_type2t::get_width() const
{
  return width * 8;
}

unsigned int string_type2t::get_length() const
{
  return width;
}

const std::vector<type2tc> &struct_union_data::get_structure_members() const
{
  return members;
}

const std::vector<irep_idt> &
struct_union_data::get_structure_member_names() const
{
  return member_names;
}

const irep_idt &struct_union_data::get_structure_name() const
{
  return name;
}

unsigned int struct_union_data::get_component_number(const irep_idt &comp) const
{
  unsigned int i = 0, count = 0, pos = 0;
  for(auto const &it : member_names)
  {
    if(it == comp)
    {
      pos = i;
      ++count;
    }
    i++;
  }

  if(count == 1)
    return pos;

  if(!count)
  {
    assert(
      0 &&
      fmt::format(
        "Looking up index of nonexistant member \"{}\" in struct/union \"{}\"",
        comp,
        name)
        .c_str());
  }
  else if(count > 1)
  {
    assert(
      0 &&
      fmt::format(
        "Name \"{}\" matches more than one member\" in struct/union \"{}\"",
        comp,
        name)
        .c_str());
  }

  abort();
}

namespace esbmct
{
template <typename... Args>
template <typename derived>
auto type2t_traits<Args...>::make_contained(typename Args::result_type... args)
  -> irep_container<base2t>
{
  return irep_container<base2t>(new derived(args...));
}
} // namespace esbmct

/*************************** Base expr2t definitions **************************/

expr2t::expr2t(const type2tc &_type, expr_ids id)
  : std::enable_shared_from_this<expr2t>(), expr_id(id), type(_type), crc_val(0)
{
}

expr2t::expr2t(const expr2t &ref)
  : std::enable_shared_from_this<expr2t>(),
    expr_id(ref.expr_id),
    type(ref.type),
    crc_val(ref.crc_val)
{
}

bool expr2t::operator==(const expr2t &ref) const
{
  if(!expr2t::cmp(ref))
    return false;

  return cmp(ref);
}

bool expr2t::operator!=(const expr2t &ref) const
{
  return !(*this == ref);
}

bool expr2t::operator<(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if(tmp < 0)
    return true;
  else if(tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

unsigned long expr2t::depth() const
{
  unsigned long num_nodes = 0;

  for(unsigned int idx = 0; idx < get_num_sub_exprs(); idx++)
  {
    const expr2tc *e = get_sub_expr(idx);
    if(is_nil_expr(*e))
      continue;
    unsigned long tmp = (*e)->depth();
    num_nodes = std::max(num_nodes, tmp);
  }

  num_nodes++; // Count ourselves.
  return num_nodes;
}

unsigned long expr2t::num_nodes() const
{
  unsigned long count = 0;

  for(unsigned int idx = 0; idx < get_num_sub_exprs(); idx++)
  {
    const expr2tc *e = get_sub_expr(idx);
    if(is_nil_expr(*e))
      continue;
    count += (*e)->num_nodes();
  }

  count++; // Count ourselves.
  return count;
}

int expr2t::ltchecked(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if(tmp != 0)
    return tmp;

  return lt(ref);
}

bool expr2t::cmp(const expr2t &ref) const
{
  if(expr_id != ref.expr_id)
    return false;

  if(type != ref.type)
    return false;

  return true;
}

int expr2t::lt(const expr2t &ref) const
{
  if(expr_id < ref.expr_id)
    return -1;
  if(expr_id > ref.expr_id)
    return 1;

  return type->ltchecked(*ref.type.get());
}

size_t expr2t::crc() const
{
  return do_crc();
}

size_t expr2t::do_crc() const
{
  boost::hash_combine(this->crc_val, type->do_crc());
  boost::hash_combine(this->crc_val, (uint8_t)expr_id);
  return this->crc_val;
}

void expr2t::hash(crypto_hash &hash) const
{
  static_assert(expr2t::end_expr_id < 256, "Expr id overflow");
  uint8_t eid = expr_id;
  hash.ingest(&eid, sizeof(eid));
  type->hash(hash);
}

static const char *expr_names[] = {
  "constant_int",
  "constant_fixedbv",
  "constant_floatbv",
  "constant_bool",
  "constant_string",
  "constant_struct",
  "constant_union",
  "constant_array",
  "constant_array_of",
  "symbol",
  "typecast",
  "bitcast",
  "nearbyint",
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
  "ieee_add",
  "ieee_sub",
  "ieee_mul",
  "ieee_div",
  "ieee_fma",
  "ieee_sqrt",
  "popcount",
  "bswap",
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
  "code_dead",
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
  "isfinite",
  "signbit",
  "concat",
  "extract",
};
// If this fires, you've added/removed an expr id, and need to update the list
// above (which is ordered according to the enum list)
static_assert(
  sizeof(expr_names) == (expr2t::end_expr_id * sizeof(char *)),
  "Missing expr name");

std::string get_expr_id(const expr2t &expr)
{
  return std::string(expr_names[expr.expr_id]);
}

std::string expr2t::pretty(unsigned int indent) const
{
  std::string ret =
    pretty_print_func<const expr2t &>(indent, expr_names[expr_id], *this);
  // Dump the type on the end.
  ret += std::string("\n") + indent_str(indent) +
         "* type : " + type->pretty(indent + 2);
  return ret;
}

void expr2t::dump() const
{
  default_message msg;
  msg.debug(pretty(0));
}

// Map a base type to it's list of names
template <typename T>
class base_to_names;

template <>
class base_to_names<type2t>
{
public:
  static constexpr const char **names = type_names;
};

template <>
class base_to_names<expr2t>
{
public:
  static constexpr const char **names = expr_names;
};

// Undoubtedly a better way of doing this...
namespace esbmct
{
template <typename... Args>
template <typename derived>
auto expr2t_traits<Args...>::make_contained(
  const type2tc &type,
  typename Args::result_type... args) -> irep_container<base2t>
{
  return irep_container<base2t>(new derived(type, args...));
}

template <typename... Args>
template <typename derived>
auto expr2t_traits_notype<Args...>::make_contained(
  typename Args::result_type... args) -> irep_container<base2t>
{
  return irep_container<base2t>(new derived(args...));
}

template <typename... Args>
template <typename derived>
auto expr2t_traits_always_construct<Args...>::make_contained(
  typename Args::result_type... args) -> irep_container<base2t>
{
  return irep_container<base2t>(new derived(args...));
}
} // namespace esbmct

/**************************** Expression constructors *************************/

unsigned long constant_int2t::as_ulong() const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  assert(!value.is_negative());
  return value.to_uint64();
}

long constant_int2t::as_long() const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  return value.to_int64();
}

bool constant_bool2t::is_true() const
{
  return value;
}

bool constant_bool2t::is_false() const
{
  return !value;
}

std::string symbol_data::get_symbol_name() const
{
  switch(rlevel)
  {
  case level0:
    return thename.as_string();
  case level1:
    return thename.as_string() + "?" + i2string(level1_num) + "!" +
           i2string(thread_num);
  case level2:
    return thename.as_string() + "?" + i2string(level1_num) + "!" +
           i2string(thread_num) + "&" + i2string(node_num) + "#" +
           i2string(level2_num);
  case level1_global:
    // Just return global name,
    return thename.as_string();
  case level2_global:
    // Global name with l2 details
    return thename.as_string() + "&" + i2string(node_num) + "#" +
           i2string(level2_num);
  default:
    assert(0 && "Unrecognized renaming level enum");
    abort();
  }
}

expr2tc constant_string2t::to_array() const
{
  std::vector<expr2tc> contents;
  unsigned int length = value.as_string().size(), i;

  type2tc type = get_uint8_type();

  for(i = 0; i < length; i++)
  {
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

const expr2tc &object_descriptor2t::get_root_object() const
{
  const expr2tc *tmp = &object;

  do
  {
    if(is_member2t(*tmp))
      tmp = &to_member2t(*tmp).source_value;
    else if(is_index2t(*tmp))
      tmp = &to_index2t(*tmp).source_value;
    else
      return *tmp;
  } while(1);
}

// For CRCing to actually be accurate, expr/type ids mustn't overflow out of
// a byte. If this happens then a) there are too many exprs, and b) the expr
// crcing code has to change.
static_assert(type2t::end_type_id <= 256, "Type id overflow");
static_assert(expr2t::end_expr_id <= 256, "Expr id overflow");

static inline std::string type_to_string(const bool &thebool, int)
{
  return (thebool) ? "true" : "false";
}

static inline std::string
type_to_string(const sideeffect_data::allockind &data, int)
{
  return (data == sideeffect_data::allockind::malloc)          ? "malloc"
         : (data == sideeffect_data::allockind::realloc)       ? "realloc"
         : (data == sideeffect_data::allockind::alloca)        ? "alloca"
         : (data == sideeffect_data::allockind::cpp_new)       ? "cpp_new"
         : (data == sideeffect_data::allockind::cpp_new_arr)   ? "cpp_new_arr"
         : (data == sideeffect_data::allockind::nondet)        ? "nondet"
         : (data == sideeffect_data::allockind::va_arg)        ? "va_arg"
         : (data == sideeffect_data::allockind::function_call) ? "function_call"
                                                               : "unknown";
}

static inline std::string type_to_string(const unsigned int &theval, int)
{
  char buffer[64];
  snprintf(buffer, 63, "%d", theval);
  return std::string(buffer);
}

static inline std::string
type_to_string(const symbol_data::renaming_level &theval, int)
{
  switch(theval)
  {
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
    assert(0 && "Unrecognized renaming level enum");
    abort();
  }
}

static inline std::string type_to_string(const BigInt &theint, int)
{
  char buffer[256], *buf;

  buf = theint.as_string(buffer, 256);
  return std::string(buf);
}

static inline std::string type_to_string(const fixedbvt &theval, int)
{
  return theval.to_ansi_c_string();
}

static inline std::string type_to_string(const ieee_floatt &theval, int)
{
  return theval.to_ansi_c_string();
}

static inline std::string
type_to_string(const std::vector<expr2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  for(auto const &it : theval)
  {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " +
               it->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

static inline std::string
type_to_string(const std::vector<type2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  for(auto const &it : theval)
  {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " +
               it->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

static inline std::string
type_to_string(const std::vector<irep_idt> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  for(auto const &it : theval)
  {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring +=
      indent_str(indent) + std::string(buffer) + ": " + it.as_string() + "\n";
    i++;
  }

  return astring;
}

static inline std::string type_to_string(const expr2tc &theval, int indent)
{
  if(theval.get() != nullptr)
    return theval->pretty(indent + 2);
  return "";
}

static inline std::string type_to_string(const type2tc &theval, int indent)
{
  if(theval.get() != nullptr)
    return theval->pretty(indent + 2);
  else
    return "";
}

static inline std::string type_to_string(const irep_idt &theval, int)
{
  return theval.as_string();
}

static inline bool do_type_cmp(const bool &side1, const bool &side2)
{
  return (side1 == side2) ? true : false;
}

static inline bool
do_type_cmp(const unsigned int &side1, const unsigned int &side2)
{
  return (side1 == side2) ? true : false;
}

static inline bool do_type_cmp(
  const sideeffect_data::allockind &side1,
  const sideeffect_data::allockind &side2)
{
  return (side1 == side2) ? true : false;
}

static inline bool do_type_cmp(
  const symbol_data::renaming_level &side1,
  const symbol_data::renaming_level &side2)
{
  return (side1 == side2) ? true : false;
}

static inline bool do_type_cmp(const BigInt &side1, const BigInt &side2)
{
  // BigInt has its own equality operator.
  return (side1 == side2) ? true : false;
}

static inline bool do_type_cmp(const fixedbvt &side1, const fixedbvt &side2)
{
  return (side1 == side2) ? true : false;
}

static inline bool
do_type_cmp(const ieee_floatt &side1, const ieee_floatt &side2)
{
  return (side1 == side2) ? true : false;
}

static inline bool do_type_cmp(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2)
{
  return (side1 == side2);
}

static inline bool do_type_cmp(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2)
{
  return (side1 == side2);
}

static inline bool do_type_cmp(
  const std::vector<irep_idt> &side1,
  const std::vector<irep_idt> &side2)
{
  return (side1 == side2);
}

static inline bool do_type_cmp(const expr2tc &side1, const expr2tc &side2)
{
  if(side1.get() == side2.get())
    return true; // Catch null
  else if(side1.get() == nullptr || side2.get() == nullptr)
    return false;
  else
    return (side1 == side2);
}

static inline bool do_type_cmp(const type2tc &side1, const type2tc &side2)
{
  if(side1.get() == side2.get())
    return true; // both null ptr check
  if(side1.get() == nullptr || side2.get() == nullptr)
    return false; // One of them is null, the other isn't
  return (side1 == side2);
}

static inline bool do_type_cmp(const irep_idt &side1, const irep_idt &side2)
{
  return (side1 == side2);
}

static inline bool
do_type_cmp(const type2t::type_ids &, const type2t::type_ids &)
{
  return true; // Dummy field comparison.
}

static inline bool
do_type_cmp(const expr2t::expr_ids &, const expr2t::expr_ids &)
{
  return true; // Dummy field comparison.
}

static inline int do_type_lt(const bool &side1, const bool &side2)
{
  if(side1 < side2)
    return -1;
  else if(side2 < side1)
    return 1;
  else
    return 0;
}

static inline int
do_type_lt(const unsigned int &side1, const unsigned int &side2)
{
  if(side1 < side2)
    return -1;
  else if(side2 < side1)
    return 1;
  else
    return 0;
}

static inline int do_type_lt(
  const sideeffect_data::allockind &side1,
  const sideeffect_data::allockind &side2)
{
  if(side1 < side2)
    return -1;
  else if(side2 < side1)
    return 1;
  else
    return 0;
}

static inline int do_type_lt(
  const symbol_data::renaming_level &side1,
  const symbol_data::renaming_level &side2)
{
  if(side1 < side2)
    return -1;
  else if(side2 < side1)
    return 1;
  else
    return 0;
}

static inline int do_type_lt(const BigInt &side1, const BigInt &side2)
{
  // BigInt also has its own less than comparator.
  return side1.compare(side2);
}

static inline int do_type_lt(const fixedbvt &side1, const fixedbvt &side2)
{
  if(side1 < side2)
    return -1;
  else if(side1 > side2)
    return 1;
  return 0;
}

static inline int do_type_lt(const ieee_floatt &side1, const ieee_floatt &side2)
{
  if(side1 < side2)
    return -1;
  else if(side1 > side2)
    return 1;
  return 0;
}

static inline int
do_type_lt(const std::vector<expr2tc> &side1, const std::vector<expr2tc> &side2)
{
  int tmp = 0;
  std::vector<expr2tc>::const_iterator it2 = side2.begin();
  for(auto const &it : side1)
  {
    tmp = it->ltchecked(**it2);
    if(tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

static inline int
do_type_lt(const std::vector<type2tc> &side1, const std::vector<type2tc> &side2)
{
  if(side1.size() < side2.size())
    return -1;
  else if(side1.size() > side2.size())
    return 1;

  int tmp = 0;
  std::vector<type2tc>::const_iterator it2 = side2.begin();
  for(auto const &it : side1)
  {
    tmp = it->ltchecked(**it2);
    if(tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

static inline int do_type_lt(
  const std::vector<irep_idt> &side1,
  const std::vector<irep_idt> &side2)
{
  if(side1 < side2)
    return -1;
  else if(side2 < side1)
    return 1;
  return 0;
}

static inline int do_type_lt(const expr2tc &side1, const expr2tc &side2)
{
  if(side1.get() == side2.get())
    return 0; // Catch nulls
  else if(side1.get() == nullptr)
    return -1;
  else if(side2.get() == nullptr)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

static inline int do_type_lt(const type2tc &side1, const type2tc &side2)
{
  if(*side1.get() == *side2.get())
    return 0; // Both may be null;
  else if(side1.get() == nullptr)
    return -1;
  else if(side2.get() == nullptr)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

static inline int do_type_lt(const irep_idt &side1, const irep_idt &side2)
{
  if(side1 < side2)
    return -1;
  if(side2 < side1)
    return 1;
  return 0;
}

static inline int do_type_lt(const type2t::type_ids &, const type2t::type_ids &)
{
  return 0; // Dummy field comparison
}

static inline int do_type_lt(const expr2t::expr_ids &, const expr2t::expr_ids &)
{
  return 0; // Dummy field comparison
}

static inline size_t do_type_crc(const bool &theval)
{
  return boost::hash<bool>()(theval);
}

static inline void do_type_hash(const bool &thebool, crypto_hash &hash)
{
  if(thebool)
  {
    uint8_t tval = 1;
    hash.ingest(&tval, sizeof(tval));
  }
  else
  {
    uint8_t tval = 0;
    hash.ingest(&tval, sizeof(tval));
  }
}

static inline size_t do_type_crc(const unsigned int &theval)
{
  return boost::hash<unsigned int>()(theval);
}

static inline void do_type_hash(const unsigned int &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

static inline size_t do_type_crc(const sideeffect_data::allockind &theval)
{
  return boost::hash<uint8_t>()(theval);
}

static inline void
do_type_hash(const sideeffect_data::allockind &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

static inline size_t do_type_crc(const symbol_data::renaming_level &theval)
{
  return boost::hash<uint8_t>()(theval);
}

static inline void
do_type_hash(const symbol_data::renaming_level &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

static inline size_t do_type_crc(const BigInt &theint)
{
  if(theint.is_zero())
    return boost::hash<uint8_t>()(0);

  size_t crc = 0;
  std::array<unsigned char, 256> buffer;
  if(theint.dump(buffer.data(), buffer.size()))
  {
    for(unsigned int i = 0; i < buffer.size(); i++)
      boost::hash_combine(crc, buffer[i]);
  }
  else
  {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return crc;
}

static inline void do_type_hash(const BigInt &theint, crypto_hash &hash)
{
  // Zero has no data in bigints.
  if(theint.is_zero())
  {
    uint8_t val = 0;
    hash.ingest(&val, sizeof(val));
    return;
  }

  std::array<unsigned char, 256> buffer;
  if(theint.dump(buffer.data(), buffer.size()))
  {
    hash.ingest(buffer.data(), buffer.size());
  }
  else
  {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
}

static inline size_t do_type_crc(const fixedbvt &theval)
{
  return do_type_crc(BigInt(theval.to_ansi_c_string().c_str()));
}

static inline void do_type_hash(const fixedbvt &theval, crypto_hash &hash)
{
  do_type_hash(BigInt(theval.to_ansi_c_string().c_str()), hash);
}

static inline size_t do_type_crc(const ieee_floatt &theval)
{
  return do_type_crc(theval.pack());
}

static inline void do_type_hash(const ieee_floatt &theval, crypto_hash &hash)
{
  do_type_hash(theval.pack(), hash);
}

static inline size_t do_type_crc(const std::vector<expr2tc> &theval)
{
  size_t crc = 0;
  for(auto const &it : theval)
    boost::hash_combine(crc, it->do_crc());

  return crc;
}

static inline void
do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash)
{
  for(auto const &it : theval)
    it->hash(hash);
}

static inline size_t do_type_crc(const std::vector<type2tc> &theval)
{
  size_t crc = 0;
  for(auto const &it : theval)
    boost::hash_combine(crc, it->do_crc());

  return crc;
}

static inline void
do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash)
{
  for(auto const &it : theval)
    it->hash(hash);
}

static inline size_t do_type_crc(const std::vector<irep_idt> &theval)
{
  size_t crc = 0;
  for(auto const &it : theval)
    boost::hash_combine(crc, it.as_string());

  return crc;
}

static inline void
do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash)
{
  for(auto const &it : theval)
    hash.ingest((void *)it.as_string().c_str(), it.as_string().size());
}

static inline size_t do_type_crc(const expr2tc &theval)
{
  if(theval.get() != nullptr)
    return theval->do_crc();
  return boost::hash<uint8_t>()(0);
}

static inline void do_type_hash(const expr2tc &theval, crypto_hash &hash)
{
  if(theval.get() != nullptr)
    theval->hash(hash);
}

static inline size_t do_type_crc(const type2tc &theval)
{
  if(theval.get() != nullptr)
    return theval->do_crc();
  return boost::hash<uint8_t>()(0);
}

static inline void do_type_hash(const type2tc &theval, crypto_hash &hash)
{
  if(theval.get() != nullptr)
    theval->hash(hash);
}

static inline size_t do_type_crc(const irep_idt &theval)
{
  return boost::hash<std::string>()(theval.as_string());
}

static inline void do_type_hash(const irep_idt &theval, crypto_hash &hash)
{
  hash.ingest((void *)theval.as_string().c_str(), theval.as_string().size());
}

static inline size_t do_type_crc(const type2t::type_ids &i)
{
  return boost::hash<uint8_t>()(i);
}

static inline void do_type_hash(const type2t::type_ids &, crypto_hash &)
{
  // Dummy field crc
}

static inline size_t do_type_crc(const expr2t::expr_ids &i)
{
  return boost::hash<uint8_t>()(i);
}

static inline void do_type_hash(const expr2t::expr_ids &, crypto_hash &)
{
  // Dummy field crc
}

template <typename T>
void do_type2string(
  const T &thething,
  unsigned int idx,
  std::string (&names)[esbmct::num_type_fields],
  list_of_memberst &vec,
  unsigned int indent)
{
  vec.push_back(member_entryt(names[idx], type_to_string(thething, indent)));
}

template <>
void do_type2string<type2t::type_ids>(
  const type2t::type_ids &,
  unsigned int,
  std::string (&)[esbmct::num_type_fields],
  list_of_memberst &,
  unsigned int)
{
  // Do nothing; this is a dummy member.
}

template <>
void do_type2string<const expr2t::expr_ids>(
  const expr2t::expr_ids &,
  unsigned int,
  std::string (&)[esbmct::num_type_fields],
  list_of_memberst &,
  unsigned int)
{
  // Do nothing; this is a dummy member.
}

template <class T>
bool do_get_sub_expr(const T &, unsigned int, unsigned int &, const expr2tc *&)
{
  return false;
}

template <>
bool do_get_sub_expr<expr2tc>(
  const expr2tc &item,
  unsigned int idx,
  unsigned int &it,
  const expr2tc *&ptr)
{
  if(idx == it)
  {
    ptr = &item;
    return true;
  }
  else
  {
    it++;
    return false;
  }
}

template <>
bool do_get_sub_expr<std::vector<expr2tc>>(
  const std::vector<expr2tc> &item,
  unsigned int idx,
  unsigned int &it,
  const expr2tc *&ptr)
{
  if(idx < it + item.size())
  {
    ptr = &item[idx - it];
    return true;
  }
  else
  {
    it += item.size();
    return false;
  }
}

// Non-const versions of the above.

template <class T>
bool do_get_sub_expr_nc(T &, unsigned int, unsigned int &, expr2tc *&)
{
  return false;
}

template <>
bool do_get_sub_expr_nc<expr2tc>(
  expr2tc &item,
  unsigned int idx,
  unsigned int &it,
  expr2tc *&ptr)
{
  if(idx == it)
  {
    ptr = &item;
    return true;
  }
  else
  {
    it++;
    return false;
  }
}

template <>
bool do_get_sub_expr_nc<std::vector<expr2tc>>(
  std::vector<expr2tc> &item,
  unsigned int idx,
  unsigned int &it,
  expr2tc *&ptr)
{
  if(idx < it + item.size())
  {
    ptr = &item[idx - it];
    return true;
  }
  else
  {
    it += item.size();
    return false;
  }
}

template <class T>
unsigned int do_count_sub_exprs(T &)
{
  return 0;
}

template <>
unsigned int do_count_sub_exprs<const expr2tc>(const expr2tc &)
{
  return 1;
}

template <>
unsigned int
do_count_sub_exprs<const std::vector<expr2tc>>(const std::vector<expr2tc> &item)
{
  return item.size();
}

// Local template for implementing delegate calling, with type dependency.
// Can't easily extend to cover types because field type is _already_ abstracted
template <typename T, typename U>
void call_expr_delegate(T &ref, U &f)
{
  // Don't do anything normally.
  (void)ref;
  (void)f;
}

template <>
void call_expr_delegate<const expr2tc, expr2t::const_op_delegate>(
  const expr2tc &ref,
  expr2t::const_op_delegate &f)
{
  f(ref);
}

template <>
void call_expr_delegate<expr2tc, expr2t::op_delegate>(
  expr2tc &ref,
  expr2t::op_delegate &f)
{
  f(ref);
}

template <>
void call_expr_delegate<const std::vector<expr2tc>, expr2t::const_op_delegate>(
  const std::vector<expr2tc> &ref,
  expr2t::const_op_delegate &f)
{
  for(const expr2tc &r : ref)
    f(r);
}

template <>
void call_expr_delegate<std::vector<expr2tc>, expr2t::op_delegate>(
  std::vector<expr2tc> &ref,
  expr2t::op_delegate &f)
{
  for(expr2tc &r : ref)
    f(r);
}

// Repeat of call_expr_delegate, but for types
template <typename T, typename U>
void call_type_delegate(T &ref, U &f)
{
  // Don't do anything normally.
  (void)ref;
  (void)f;
}

template <>
void call_type_delegate<const type2tc, type2t::const_subtype_delegate>(
  const type2tc &ref,
  type2t::const_subtype_delegate &f)
{
  f(ref);
}

template <>
void call_type_delegate<type2tc, type2t::subtype_delegate>(
  type2tc &ref,
  type2t::subtype_delegate &f)
{
  f(ref);
}

template <>
void call_type_delegate<
  const std::vector<type2tc>,
  type2t::const_subtype_delegate>(
  const std::vector<type2tc> &ref,
  type2t::const_subtype_delegate &f)
{
  for(const type2tc &r : ref)
    f(r);
}

template <>
void call_type_delegate<std::vector<type2tc>, type2t::subtype_delegate>(
  std::vector<type2tc> &ref,
  type2t::subtype_delegate &f)
{
  for(type2tc &r : ref)
    f(r);
}

/************************ Second attempt at irep templates ********************/

// Implementations of common methods, recursively.

// Top level type method definition (above recursive def)
// exprs

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
const expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::
  get_sub_expr(unsigned int i) const
{
  return superclass::get_sub_expr_rec(0, i); // Skips expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::
  get_sub_expr_nc(unsigned int i)
{
  return superclass::get_sub_expr_nc_rec(0, i); // Skips expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
unsigned int
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::
  get_num_sub_exprs() const
{
  return superclass::get_num_sub_exprs_rec(); // Skips expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  expr_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_operand_impl_const(expr2t::const_op_delegate &f) const
{
  superclass::foreach_operand_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  expr_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_operand_impl(expr2t::op_delegate &f)
{
  superclass::foreach_operand_impl_rec(f);
}

// Types

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  type_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_subtype_impl_const(type2t::const_subtype_delegate &f) const
{
  superclass::foreach_subtype_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  type_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_subtype_impl(type2t::subtype_delegate &f)
{
  superclass::foreach_subtype_impl_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
auto esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::clone()
    const -> base_container2tc
{
  const derived *derived_this = static_cast<const derived *>(this);
  // Use std::make_shared to clone this with one allocation, it puts the ref
  // counting block ahead of the data object itself. This necessitates making
  // a bare std::shared_ptr first, and then feeding that into an expr2tc
  // container.
  // Generally, storing an irep in a bare std::shared_ptr loses the detach
  // facility and breaks everything, this is an exception.
  return base_container2tc(std::make_shared<derived>(*derived_this));
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
list_of_memberst
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::
  tostring(unsigned int indent) const
{
  list_of_memberst thevector;

  superclass::tostring_rec(0, thevector, indent); // Skips type_id / expr_id
  return thevector;
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
bool esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::cmp(
    const base2t &ref) const
{
  return cmp_rec(ref); // _includes_ type_id / expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
int esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::lt(
    const base2t &ref) const
{
  return lt_rec(ref); // _includes_ type_id / expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
size_t
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::
  do_crc() const
{
  if(this->crc_val != 0)
    return this->crc_val;

  // Starting from 0, pass a crc value through all the sub-fields of this
  // expression. Store it into crc_val.
  assert(this->crc_val == 0);

  do_crc_rec(); // _includes_ type_id / expr_id

  // Finally, combine the crc of this expr with the input , and return
  return this->crc_val;
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::hash(
    crypto_hash &hash) const
{
  hash_rec(hash); // _includes_ type_id / expr_id
}

// The, *actual* recursive defs

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    tostring_rec(unsigned int idx, list_of_memberst &vec, unsigned int indent)
      const
{
  // Skip over type fields in expressions. Alas, this is a design oversight,
  // without this we would screw up the field name list.
  // It escapes me why this isn't printed here anyway, it gets printed in the
  // end.
  if(
    std::is_same<cur_type, type2tc>::value &&
    std::is_base_of<expr2t, derived>::value)
  {
    superclass::tostring_rec(idx, vec, indent);
    return;
  }

  // Insert our particular member to string list.
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;
  do_type2string<cur_type>(
    derived_this->*m_ptr, idx, derived_this->field_names, vec, indent);

  // Recurse
  superclass::tostring_rec(idx + 1, vec, indent);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
bool esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::cmp_rec(
    const base2t &ref) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  if(!do_type_cmp(derived_this->*m_ptr, ref2->*m_ptr))
    return false;

  return superclass::cmp_rec(ref);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
int esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::lt_rec(
    const base2t &ref) const
{
  int tmp;
  const derived *derived_this = static_cast<const derived *>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  tmp = do_type_lt(derived_this->*m_ptr, ref2->*m_ptr);
  if(tmp != 0)
    return tmp;

  return superclass::lt_rec(ref);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    do_crc_rec() const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  size_t tmp = do_type_crc(derived_this->*m_ptr);
  boost::hash_combine(this->crc_val, tmp);

  superclass::do_crc_rec();
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    hash_rec(crypto_hash &hash) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;
  do_type_hash(derived_this->*m_ptr, hash);

  superclass::hash_rec(hash);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
const expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::
  get_sub_expr_rec(unsigned int cur_idx, unsigned int desired) const
{
  const expr2tc *ptr;
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if(do_get_sub_expr(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_rec(cur_idx, desired);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::
  get_sub_expr_nc_rec(unsigned int cur_idx, unsigned int desired)
{
  expr2tc *ptr;
  derived *derived_this = static_cast<derived *>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if(do_get_sub_expr_nc(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_nc_rec(cur_idx, desired);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
unsigned int
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::
  get_num_sub_exprs_rec() const
{
  unsigned int num = 0;
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  num = do_count_sub_exprs(derived_this->*m_ptr);
  return num + superclass::get_num_sub_exprs_rec();
}

// Operand iteration specialized for expr2tc: call delegate.
template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_operand_impl_const_rec(expr2t::const_op_delegate &f) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_operand_impl_rec(expr2t::op_delegate &f)
{
  derived *derived_this = static_cast<derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_rec(f);
}

// Misery cakes to add readwrite modifier for non-const fields.
namespace esbmct
{
template <class Reg, class targfield>
struct magical_mystery_modifier
{
public:
  void operator()(Reg &reg, const char *fname, targfield foo)
  {
    (void)reg;
    (void)fname;
    (void)foo;
  }
};

template <class Reg, class foo, class bar>
struct magical_mystery_modifier<Reg, foo bar::*>
{
public:
  void operator()(
    Reg &reg,
    const char *fname,
    foo bar::*baz,
    typename boost::disable_if<typename boost::is_const<foo>::type, bool>::type
      qux = false)
  {
    (void)qux;
    reg.def_readwrite(fname, baz);
  }
};

template <class Reg, class foo, class bar>
struct magical_mystery_modifier<Reg, const foo bar::*>
{
public:
  void operator()(Reg &reg, const char *fname, const foo bar::*baz)
  {
    (void)reg;
    (void)fname;
    (void)baz;
  }
};
} // namespace esbmct

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_subtype_impl_const_rec(type2t::const_subtype_delegate &f) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_type_delegate(derived_this->*m_ptr, f);

  superclass::foreach_subtype_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename container,
  typename enable,
  typename fields>
void esbmct::
  irep_methods2<derived, baseclass, traits, container, enable, fields>::
    foreach_subtype_impl_rec(type2t::subtype_delegate &f)
{
  derived *derived_this = static_cast<derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_type_delegate(derived_this->*m_ptr, f);

  superclass::foreach_subtype_impl_rec(f);
}

/********************** Constants and explicit instantiations *****************/

std::string bool_type2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string empty_type2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string symbol_type2t::field_names[esbmct::num_type_fields] =
  {"symbol_name", "", "", "", ""};
std::string struct_type2t::field_names[esbmct::num_type_fields] =
  {"members", "member_names", "member_pretty_names", "typename", "packed", ""};
std::string union_type2t::field_names[esbmct::num_type_fields] =
  {"members", "member_names", "member_pretty_names", "typename", "packed", ""};
std::string unsignedbv_type2t::field_names[esbmct::num_type_fields] =
  {"width", "", "", "", ""};
std::string signedbv_type2t::field_names[esbmct::num_type_fields] =
  {"width", "", "", "", ""};
std::string code_type2t::field_names[esbmct::num_type_fields] =
  {"arguments", "ret_type", "argument_names", "ellipsis", ""};
std::string array_type2t::field_names[esbmct::num_type_fields] =
  {"subtype", "array_size", "size_is_infinite", "", ""};
std::string pointer_type2t::field_names[esbmct::num_type_fields] =
  {"subtype", "", "", "", ""};
std::string fixedbv_type2t::field_names[esbmct::num_type_fields] =
  {"width", "integer_bits", "", "", ""};
std::string floatbv_type2t::field_names[esbmct::num_type_fields] =
  {"fraction", "exponent", "", "", ""};
std::string string_type2t::field_names[esbmct::num_type_fields] =
  {"width", "", "", "", ""};
std::string cpp_name_type2t::field_names[esbmct::num_type_fields] =
  {"name", "template args", "", "", ""};

// Exprs

std::string constant_int2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_fixedbv2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_floatbv2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_struct2t::field_names[esbmct::num_type_fields] =
  {"members", "", "", "", ""};
std::string constant_union2t::field_names[esbmct::num_type_fields] =
  {"members", "", "", "", ""};
std::string constant_bool2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_array2t::field_names[esbmct::num_type_fields] =
  {"members", "", "", "", ""};
std::string constant_array_of2t::field_names[esbmct::num_type_fields] =
  {"initializer", "", "", "", ""};
std::string constant_string2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string symbol2t::field_names[esbmct::num_type_fields] =
  {"name", "renamelev", "level1_num", "level2_num", "thread_num", "node_num"};
std::string typecast2t::field_names[esbmct::num_type_fields] =
  {"from", "rounding_mode", "", "", "", ""};
std::string bitcast2t::field_names[esbmct::num_type_fields] =
  {"from", "", "", "", ""};
std::string nearbyint2t::field_names[esbmct::num_type_fields] =
  {"from", "rounding_mode", "", "", "", ""};
std::string if2t::field_names[esbmct::num_type_fields] =
  {"cond", "true_value", "false_value", "", ""};
std::string equality2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string notequal2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string lessthan2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string greaterthan2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string lessthanequal2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string greaterthanequal2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string not2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string and2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string or2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string xor2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string implies2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitand2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitor2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitxor2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitnand2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitnor2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitnxor2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string lshr2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string bitnot2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string neg2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string abs2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string add2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string sub2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string mul2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string div2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string ieee_add2t::field_names[esbmct::num_type_fields] =
  {"rounding_mode", "side_1", "side_2", "", "", ""};
std::string ieee_sub2t::field_names[esbmct::num_type_fields] =
  {"rounding_mode", "side_1", "side_2", "", "", ""};
std::string ieee_mul2t::field_names[esbmct::num_type_fields] =
  {"rounding_mode", "side_1", "side_2", "", "", ""};
std::string ieee_div2t::field_names[esbmct::num_type_fields] =
  {"rounding_mode", "side_1", "side_2", "", "", ""};
std::string ieee_fma2t::field_names[esbmct::num_type_fields] =
  {"rounding_mode", "value_1", "value_2", "value_3", "", ""};
std::string ieee_sqrt2t::field_names[esbmct::num_type_fields] =
  {"rounding_mode", "value", "", "", ""};
std::string modulus2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string shl2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string ashr2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string same_object2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string pointer_offset2t::field_names[esbmct::num_type_fields] =
  {"pointer_obj", "", "", "", ""};
std::string pointer_object2t::field_names[esbmct::num_type_fields] =
  {"pointer_obj", "", "", "", ""};
std::string address_of2t::field_names[esbmct::num_type_fields] =
  {"pointer_obj", "", "", "", ""};
std::string byte_extract2t::field_names[esbmct::num_type_fields] =
  {"source_value", "source_offset", "big_endian", "", ""};
std::string byte_update2t::field_names[esbmct::num_type_fields] =
  {"source_value", "source_offset", "update_value", "big_endian", ""};
std::string with2t::field_names[esbmct::num_type_fields] =
  {"source_value", "update_field", "update_value", "", ""};
std::string member2t::field_names[esbmct::num_type_fields] =
  {"source_value", "member_name", "", "", ""};
std::string index2t::field_names[esbmct::num_type_fields] =
  {"source_value", "index", "", "", ""};
std::string isnan2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string overflow2t::field_names[esbmct::num_type_fields] =
  {"operand", "", "", "", ""};
std::string overflow_cast2t::field_names[esbmct::num_type_fields] =
  {"operand", "bits", "", "", ""};
std::string overflow_neg2t::field_names[esbmct::num_type_fields] =
  {"operand", "", "", "", ""};
std::string unknown2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string invalid2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string null_object2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string dynamic_object2t::field_names[esbmct::num_type_fields] =
  {"instance", "invalid", "unknown", "", ""};
std::string dereference2t::field_names[esbmct::num_type_fields] =
  {"pointer", "", "", "", ""};
std::string valid_object2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string deallocated_obj2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string dynamic_size2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string sideeffect2t::field_names[esbmct::num_type_fields] =
  {"operand", "size", "arguments", "alloctype", "kind"};
std::string code_block2t::field_names[esbmct::num_type_fields] =
  {"operands", "", "", "", ""};
std::string code_assign2t::field_names[esbmct::num_type_fields] =
  {"target", "source", "", "", ""};
std::string code_init2t::field_names[esbmct::num_type_fields] =
  {"target", "source", "", "", ""};
std::string code_decl2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_dead2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_printf2t::field_names[esbmct::num_type_fields] =
  {"operands", "", "", "", ""};
std::string code_expression2t::field_names[esbmct::num_type_fields] =
  {"operand", "", "", "", ""};
std::string code_return2t::field_names[esbmct::num_type_fields] =
  {"operand", "", "", "", ""};
std::string code_skip2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string code_free2t::field_names[esbmct::num_type_fields] =
  {"operand", "", "", "", ""};
std::string code_goto2t::field_names[esbmct::num_type_fields] =
  {"target", "", "", "", ""};
std::string object_descriptor2t::field_names[esbmct::num_type_fields] =
  {"object", "offset", "alignment", "", ""};
std::string code_function_call2t::field_names[esbmct::num_type_fields] =
  {"return_sym", "function", "operands", "", ""};
std::string code_comma2t::field_names[esbmct::num_type_fields] =
  {"side_1", "side_2", "", "", ""};
std::string invalid_pointer2t::field_names[esbmct::num_type_fields] =
  {"pointer_obj", "", "", "", ""};
std::string code_asm2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_cpp_del_array2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_cpp_delete2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_cpp_catch2t::field_names[esbmct::num_type_fields] =
  {"exception_list", "", "", "", ""};
std::string code_cpp_throw2t::field_names[esbmct::num_type_fields] =
  {"operand", "exception_list", "", "", ""};
std::string code_cpp_throw_decl2t::field_names[esbmct::num_type_fields] =
  {"exception_list", "", "", "", ""};
std::string code_cpp_throw_decl_end2t::field_names[esbmct::num_type_fields] =
  {"exception_list", "", "", "", ""};
std::string isinf2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string isnormal2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string isfinite2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string signbit2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string popcount2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string bswap2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string concat2t::field_names[esbmct::num_type_fields] =
  {"forward", "aft", "", "", ""};
std::string extract2t::field_names[esbmct::num_type_fields] =
  {"from", "upper", "lower", "", ""};

// This has become particularly un-fun with the arrival of gcc 6.x and clang
// 3.8 (roughly). Both are very aggressive wrt. whether templates are actually
// instantiated or not, and refuse to instantiate template base classes
// implicitly. Unfortunately, we relied on that before; it might still be
// instantiating some of the irep_methods2 chain, but it's not doing the
// method definitions, which leads to linking failures later.
//
// I've experimented with a variety of ways to implicitly require each method
// of our template chain, but none seem to succeed, and the compiler goes a
// long way out of it's path to avoid these instantiations. The real real
// issue seems to be virtual functions, the compiler can jump through many
// hoops to get method addresses out of the vtable, rather than having to
// implicitly define it. One potential workaround may be a non-virtual method
// that gets defined that calls all virtual methods explicitly?
//
// Anyway: the workaround is to explicitly instantiate each level of the
// irep_methods2 hierarchy, with associated pain and suffering. This means
// that our template system isn't completely variadic, you have to know how
// many levels to instantiate when you reach this level, explicitly. Which
// sucks, but is a small price to pay.

#undef irep_typedefs
#undef irep_typedefs_empty

#define irep_typedefs0(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;

#define irep_typedefs1(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;

#define irep_typedefs2(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;

#define irep_typedefs3(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;

#define irep_typedefs4(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename superclass::traits::fields>::  \
          type>::type>::type>::type>::type>;

#define irep_typedefs5(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename superclass::traits::fields>::  \
          type>::type>::type>::type>::type>;                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<boost::mpl::pop_front<                               \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename boost::mpl::pop_front<         \
          typename superclass::traits::fields>::type>::type>::type>::type>::   \
                            type>::type>;

#define irep_typedefs6(basename, superclass)                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename superclass::traits::fields>::  \
          type>::type>::type>::type>::type>;                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<boost::mpl::pop_front<                               \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename boost::mpl::pop_front<         \
          typename superclass::traits::fields>::type>::type>::type>::type>::   \
                            type>::type>;                                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc,                                                             \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<boost::mpl::pop_front<                    \
        typename boost::mpl::pop_front<typename boost::mpl::pop_front<         \
          typename boost::mpl::pop_front<typename boost::mpl::pop_front<       \
            typename superclass::traits::fields>::type>::type>::type>::type>:: \
                                       type>::type>::type>;

////////////////////////////

#define type_typedefs1(basename, superclass)                                   \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    typename superclass::traits,                                               \
    basename##2tc>;                                                            \
  irep_typedefs1(basename, superclass)

#define type_typedefs2(basename, superclass)                                   \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    typename superclass::traits,                                               \
    basename##2tc>;                                                            \
  irep_typedefs2(basename, superclass)

#define type_typedefs3(basename, superclass)                                   \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    typename superclass::traits,                                               \
    basename##2tc>;                                                            \
  irep_typedefs3(basename, superclass)

#define type_typedefs4(basename, superclass)                                   \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    typename superclass::traits,                                               \
    basename##2tc>;                                                            \
  irep_typedefs4(basename, superclass)

#define type_typedefs5(basename, superclass)                                   \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    typename superclass::traits,                                               \
    basename##2tc>;                                                            \
  irep_typedefs5(basename, superclass)

#define type_typedefs6(basename, superclass)                                   \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    typename superclass::traits,                                               \
    basename##2tc>;                                                            \
  irep_typedefs6(basename, superclass)

#define type_typedefs_empty(basename)                                          \
  template class esbmct::type_methods2<                                        \
    basename##2t,                                                              \
    type2t,                                                                    \
    esbmct::type2t_default_traits,                                             \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    type2t,                                                                    \
    esbmct::type2t_default_traits,                                             \
    basename##2tc>;

type_typedefs_empty(bool_type) type_typedefs_empty(empty_type)
  type_typedefs1(symbol_type, symbol_type_data)
    type_typedefs5(struct_type, struct_union_data)
      type_typedefs5(union_type, struct_union_data)
        type_typedefs1(unsignedbv_type, bv_data)
          type_typedefs1(signedbv_type, bv_data)
            type_typedefs4(code_type, code_data)
              type_typedefs3(array_type, array_data)
                type_typedefs1(pointer_type, pointer_data)
                  type_typedefs2(fixedbv_type, fixedbv_data)
                    type_typedefs2(floatbv_type, floatbv_data)
                      type_typedefs1(string_type, string_data)
                        type_typedefs2(cpp_name_type, cpp_name_data)

// Explicit instanciation for exprs.

#define expr_typedefs1(basename, superclass)                                   \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  irep_typedefs1(basename, superclass)

#define expr_typedefs2(basename, superclass)                                   \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  irep_typedefs2(basename, superclass)

#define expr_typedefs3(basename, superclass)                                   \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  irep_typedefs3(basename, superclass)

#define expr_typedefs4(basename, superclass)                                   \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  irep_typedefs4(basename, superclass)

#define expr_typedefs5(basename, superclass)                                   \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  irep_typedefs5(basename, superclass)

#define expr_typedefs6(basename, superclass)                                   \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  irep_typedefs6(basename, superclass)

#define expr_typedefs_empty(basename, superclass)                              \
  template class esbmct::expr_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    esbmct::expr2t_default_traits,                                             \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    basename##2tc>;                                                            \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    esbmct::expr2t_default_traits,                                             \
    basename##2tc,                                                             \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;

                          expr_typedefs1(constant_int, constant_int_data);
expr_typedefs1(constant_fixedbv, constant_fixedbv_data);
expr_typedefs1(constant_floatbv, constant_floatbv_data);
expr_typedefs1(constant_struct, constant_datatype_data);
expr_typedefs1(constant_union, constant_datatype_data);
expr_typedefs1(constant_array, constant_datatype_data);
expr_typedefs1(constant_bool, constant_bool_data);
expr_typedefs1(constant_array_of, constant_array_of_data);
expr_typedefs1(constant_string, constant_string_data);
expr_typedefs6(symbol, symbol_data);
expr_typedefs2(nearbyint, typecast_data);
expr_typedefs2(typecast, typecast_data);
expr_typedefs2(bitcast, bitcast_data);
expr_typedefs3(if, if_data);
expr_typedefs2(equality, relation_data);
expr_typedefs2(notequal, relation_data);
expr_typedefs2(lessthan, relation_data);
expr_typedefs2(greaterthan, relation_data);
expr_typedefs2(lessthanequal, relation_data);
expr_typedefs2(greaterthanequal, relation_data);
expr_typedefs1(not, bool_1op);
expr_typedefs2(and, logic_2ops);
expr_typedefs2(or, logic_2ops);
expr_typedefs2(xor, logic_2ops);
expr_typedefs2(implies, logic_2ops);
expr_typedefs2(bitand, bit_2ops);
expr_typedefs2(bitor, bit_2ops);
expr_typedefs2(bitxor, bit_2ops);
expr_typedefs2(bitnand, bit_2ops);
expr_typedefs2(bitnor, bit_2ops);
expr_typedefs2(bitnxor, bit_2ops);
expr_typedefs2(lshr, bit_2ops);
expr_typedefs1(bitnot, bitnot_data);
expr_typedefs1(neg, arith_1op);
expr_typedefs1(abs, arith_1op);
expr_typedefs2(add, arith_2ops);
expr_typedefs2(sub, arith_2ops);
expr_typedefs2(mul, arith_2ops);
expr_typedefs2(div, arith_2ops);
expr_typedefs3(ieee_add, ieee_arith_2ops);
expr_typedefs3(ieee_sub, ieee_arith_2ops);
expr_typedefs3(ieee_mul, ieee_arith_2ops);
expr_typedefs3(ieee_div, ieee_arith_2ops);
expr_typedefs4(ieee_fma, ieee_arith_3ops);
expr_typedefs2(ieee_sqrt, ieee_arith_1op);
expr_typedefs2(modulus, arith_2ops);
expr_typedefs2(shl, arith_2ops);
expr_typedefs2(ashr, arith_2ops);
expr_typedefs2(same_object, same_object_data);
expr_typedefs1(pointer_offset, pointer_ops);
expr_typedefs1(pointer_object, pointer_ops);
expr_typedefs1(address_of, pointer_ops);
expr_typedefs3(byte_extract, byte_extract_data);
expr_typedefs4(byte_update, byte_update_data);
expr_typedefs3(with, with_data);
expr_typedefs2(member, member_data);
expr_typedefs2(index, index_data);
expr_typedefs1(isnan, bool_1op);
expr_typedefs1(overflow, overflow_ops);
expr_typedefs2(overflow_cast, overflow_cast_data);
expr_typedefs1(overflow_neg, overflow_ops);
expr_typedefs_empty(unknown, expr2t);
expr_typedefs_empty(invalid, expr2t);
expr_typedefs_empty(null_object, expr2t);
expr_typedefs3(dynamic_object, dynamic_object_data);
expr_typedefs2(dereference, dereference_data);
expr_typedefs1(valid_object, object_ops);
expr_typedefs1(deallocated_obj, object_ops);
expr_typedefs1(dynamic_size, object_ops);
expr_typedefs5(sideeffect, sideeffect_data);
expr_typedefs1(code_block, code_block_data);
expr_typedefs2(code_assign, code_assign_data);
expr_typedefs2(code_init, code_assign_data);
expr_typedefs1(code_decl, code_decl_data);
expr_typedefs1(code_dead, code_decl_data);
expr_typedefs1(code_printf, code_printf_data);
expr_typedefs1(code_expression, code_expression_data);
expr_typedefs1(code_return, code_expression_data);
expr_typedefs_empty(code_skip, expr2t);
expr_typedefs1(code_free, code_expression_data);
expr_typedefs1(code_goto, code_goto_data);
expr_typedefs3(object_descriptor, object_desc_data);
expr_typedefs3(code_function_call, code_funccall_data);
expr_typedefs2(code_comma, code_comma_data);
expr_typedefs1(invalid_pointer, invalid_pointer_ops);
expr_typedefs1(code_asm, code_asm_data);
expr_typedefs1(code_cpp_del_array, code_expression_data);
expr_typedefs1(code_cpp_delete, code_expression_data);
expr_typedefs1(code_cpp_catch, code_cpp_catch_data);
expr_typedefs2(code_cpp_throw, code_cpp_throw_data);
expr_typedefs2(code_cpp_throw_decl, code_cpp_throw_decl_data);
expr_typedefs1(code_cpp_throw_decl_end, code_cpp_throw_decl_data);
expr_typedefs1(isinf, bool_1op);
expr_typedefs1(isnormal, bool_1op);
expr_typedefs1(isfinite, bool_1op);
expr_typedefs1(signbit, overflow_ops);
expr_typedefs1(popcount, overflow_ops);
expr_typedefs1(bswap, arith_1op);
expr_typedefs2(concat, bit_2ops);
expr_typedefs3(extract, extract_data);
