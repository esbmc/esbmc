#include <stdarg.h>
#include <string.h>

#include "irep2.h"

#include <solvers/prop/prop_conv.h>

template <class T>
list_of_memberst
tostring_func(const char *name, const T *val, ...)
{
  va_list list;

  list_of_memberst thevector;

  std::string stringval = (*val)->pretty(2);
  thevector.push_back(std::pair<std::string,std::string>
                               (std::string(name), stringval));

  va_start(list, val);
  do {
    const char *listname = va_arg(list, const char *);
    if (strlen(listname) == 0)
      return thevector;

    const T *v2 = va_arg(list, const T *);
    stringval = (*v2)->pretty(2);
    thevector.push_back(std::pair<std::string,std::string>
                                 (std::string(name), stringval));
  } while (1);
}

/*************************** Base type2t definitions **************************/

type2t::type2t(type_ids id)
  : type_id(id)
{
}

bool
type2t::operator==(const type2t &ref) const
{

  return cmp(ref);
}

bool
type2t::operator!=(const type2t &ref) const
{

  return !(*this == ref);
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
type2t::cmp(const type2t &ref) const
{

  if (type_id == ref.type_id)
    return true;
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

template<class derived>
void
type_body<derived>::convert_smt_type(const prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

template<class derived>
void
bv_type_body<derived>::convert_smt_type(const prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

template<class derived>
void
struct_union_type_body2t<derived>::convert_smt_type(const prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

bv_type2t::bv_type2t(type2t::type_ids id, unsigned int _width)
  : type_body<bv_type2t>(id),
    width(_width)
{
}

unsigned int
bv_type2t::get_width(void) const
{
  return width;
}

bool
bv_type2t::cmp(const bv_type2t &ref) const
{
  if (width == ref.width)
    return true;
  return false;
}

int
bv_type2t::lt(const type2t &ref) const
{
  const bv_type2t &ref2 = static_cast<const bv_type2t &>(ref);

  if (width < ref2.width)
    return -1;
  if (width > ref2.width)
    return 1;

  return 0;
}

list_of_memberst
bv_type2t::tostring(void) const
{
  char bees[256];
  list_of_memberst membs;

  snprintf(bees, 255, "%d", width);
  bees[255] = '\0';
  membs.push_back(member_entryt("width", bees));
  return membs;
}

struct_union_type2t::struct_union_type2t(type_ids id,
                                         const std::vector<type2tc> &_members,
                                         std::vector<std::string> memb_names,
                                         std::string _name)
  : type_body<struct_union_type2t>(id), members(_members),
                                        member_names(memb_names),
                                        name(_name)
{
}

bool
struct_union_type2t::cmp(const struct_union_type2t &ref) const
{

  if (name != ref.name)
    return false;

  if (members.size() != ref.members.size())
    return false;

  if (members != ref.members)
    return false;

  if (member_names != ref.member_names)
    return false;

  return true;
}

int
struct_union_type2t::lt(const type2t &ref) const
{
  const struct_union_type2t &ref2 =
                            static_cast<const struct_union_type2t &>(ref);

  if (name < ref2.name)
    return -1;
  if (name > ref2.name)
    return 1;

  if (members.size() < ref2.members.size())
    return -1;
  if (members.size() > ref2.members.size())
    return 1;

  if (members < ref2.members)
    return -1;
  if (members > ref2.members)
    return 1;

  if (member_names < ref2.member_names)
    return -1;
  if (member_names > ref2.member_names)
    return 1;

  return 0;
}

list_of_memberst
struct_union_type2t::tostring(void) const
{
  char bees[256];
  list_of_memberst membs;

  membs.push_back(member_entryt("struct name", name));

  unsigned int i = 0;
  forall_types(it, members) {
    snprintf(bees, 255, "member \"%s\" (%d)", member_names[i].c_str(), i);
    bees[255] = '\0';
    membs.push_back(member_entryt(std::string(bees), (*it)->pretty(2)));
  }

  return membs;
}

bool_type2t::bool_type2t(void)
  : type_body<bool_type2t>(bool_id)
{
}

unsigned int
bool_type2t::get_width(void) const
{
  return 1;
}

bool
bool_type2t::cmp(const bool_type2t &ref __attribute__((unused))) const
{
  return true; // No data stored in bool type
}

int
bool_type2t::lt(const type2t &ref __attribute__((unused))) const
{
  return 0; // No data stored in bool type
}

list_of_memberst
bool_type2t::tostring(void) const
{
  return list_of_memberst(); // No data here
}

signedbv_type2t::signedbv_type2t(unsigned int width)
  : bv_type_body<signedbv_type2t>(signedbv_id, width)
{
}

bool
signedbv_type2t::cmp(const signedbv_type2t &ref) const
{
  return bv_type2t::cmp(ref);
}

int
signedbv_type2t::lt(const type2t &ref) const
{
  return bv_type2t::lt(ref);
}

unsignedbv_type2t::unsignedbv_type2t(unsigned int width)
  : bv_type_body<unsignedbv_type2t>(unsignedbv_id, width)
{
}

bool
unsignedbv_type2t::cmp(const unsignedbv_type2t &ref) const
{
  return bv_type2t::cmp(ref);
}

int
unsignedbv_type2t::lt(const type2t &ref) const
{
  return bv_type2t::lt(ref);
}

array_type2t::array_type2t(const type2tc t, const expr2tc s, bool inf)
  : type_body<array_type2t>(array_id), subtype(t), array_size(s),
    size_is_infinite(inf)
{
}

unsigned int
array_type2t::get_width(void) const
{
  // Two edge cases: the array can have infinite size, or it can have a dynamic
  // size that's determined by the solver.
  if (size_is_infinite)
    throw new inf_sized_array_excp();

  if (array_size->expr_id != expr2t::constant_int_id)
    throw new dyn_sized_array_excp();

  // Otherwise, we can multiply the size of the subtype by the number of elements.
  unsigned int sub_width = subtype->get_width();

  expr2t *elem_size = array_size.get();
  constant_int2t *const_elem_size = dynamic_cast<constant_int2t*>(elem_size);
  assert(const_elem_size != NULL);
  unsigned long num_elems = const_elem_size->as_ulong();

  return num_elems * sub_width;
}

bool
array_type2t::cmp(const array_type2t &ref) const
{

  // Check subtype type matches
  if (subtype != ref.subtype)
    return false;

  // If both sizes are infinite, we're the same type
  if (size_is_infinite && ref.size_is_infinite)
    return true;

  // If only one size is infinite, then we're not the same, and liable to step
  // on a null pointer if we access array_size.
  if (size_is_infinite || ref.size_is_infinite)
    return false;

  // Otherwise,
  return (array_size == ref.array_size);
}

int
array_type2t::lt(const type2t &ref) const
{
  const array_type2t &ref2 = static_cast<const array_type2t&>(ref);

  int tmp = subtype->ltchecked(*ref2.subtype.get());
  if (tmp != 0)
    return tmp;

  if (size_is_infinite < ref2.size_is_infinite)
    return -1;
  if (size_is_infinite > ref2.size_is_infinite)
    return 1;

  return array_size->ltchecked(*ref2.array_size.get());
}

pointer_type2t::pointer_type2t(type2tc _sub)
  : type_body<pointer_type2t>(pointer_id), subtype(_sub)
{
}

unsigned int
pointer_type2t::get_width(void) const
{
  return config.ansi_c.pointer_width;
}

bool
pointer_type2t::cmp(const pointer_type2t &ref) const
{

  return subtype == ref.subtype;
}

int
pointer_type2t::lt(const type2t &ref) const
{
  const pointer_type2t &ref2 = static_cast<const pointer_type2t&>(ref);

  return subtype->ltchecked(*ref2.subtype.get());
}

empty_type2t::empty_type2t(void)
  : type_body<empty_type2t>(empty_id)
{
}

unsigned int
empty_type2t::get_width(void) const
{
  assert(0 && "Fetching width of empty type - invalid operation");
}

bool
empty_type2t::cmp(const empty_type2t &ref __attribute__((unused))) const
{
  return true; // Two empty types always compare true.
}

int
empty_type2t::lt(const type2t &ref __attribute__((unused))) const
{
  return 0; // Two empty types always same.
}

list_of_memberst
empty_type2t::tostring(void) const
{
  return list_of_memberst(); // No data here
}

symbol_type2t::symbol_type2t(const dstring sym_name)
  : type_body<symbol_type2t>(symbol_id), symbol_name(sym_name)
{
}

unsigned int
symbol_type2t::get_width(void) const
{
  assert(0 && "Fetching width of symbol type - invalid operation");
}

bool
symbol_type2t::cmp(const symbol_type2t &ref) const
{
  return symbol_name == ref.symbol_name;
}

int
symbol_type2t::lt(const type2t &ref) const
{
  const symbol_type2t &ref2 = static_cast<const symbol_type2t &>(ref);

  if (symbol_name < ref2.symbol_name)
    return -1;
  if (ref2.symbol_name < symbol_name)
    return 1;
  return 0;
}

list_of_memberst
symbol_type2t::tostring(void) const
{
  list_of_memberst membs;
  membs.push_back(member_entryt("symbol", symbol_name.as_string()));
  return membs;
}

struct_type2t::struct_type2t(std::vector<type2tc> &members,
                             std::vector<std::string> memb_names,
                             std::string name)
  : struct_union_type_body2t<struct_type2t>(struct_id, members, memb_names, name)
{
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

bool
struct_type2t::cmp(const struct_type2t &ref) const
{

  return struct_union_type2t::cmp(ref);
}

int
struct_type2t::lt(const type2t &ref) const
{

  return struct_union_type2t::lt(ref);
}

list_of_memberst
struct_type2t::tostring(void) const
{
  return struct_union_type2t::tostring();
}

union_type2t::union_type2t(std::vector<type2tc> &members,
                           std::vector<std::string> memb_names,
                           std::string name)
  : struct_union_type_body2t<union_type2t>(union_id, members, memb_names, name)
{
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

bool
union_type2t::cmp(const union_type2t &ref) const
{

  return struct_union_type2t::cmp(ref);
}

int
union_type2t::lt(const type2t &ref) const
{

  return struct_union_type2t::lt(ref);
}

list_of_memberst
union_type2t::tostring(void) const
{
  return struct_union_type2t::tostring();
}

fixedbv_type2t::fixedbv_type2t(unsigned int _width, unsigned int integer)
  : type_body<fixedbv_type2t>(fixedbv_id), width(_width),
                                           integer_bits(integer)
{
}

unsigned int
fixedbv_type2t::get_width(void) const
{
  return width;
}

bool
fixedbv_type2t::cmp(const fixedbv_type2t &ref) const
{

  if (width != ref.width)
    return false;

  if (integer_bits != ref.integer_bits)
    return false;

  return true;
}

int
fixedbv_type2t::lt(const type2t &ref) const
{
  const fixedbv_type2t &ref2 = static_cast<const fixedbv_type2t &>(ref);

  if (width < ref2.width)
    return -1;
  if (width > ref2.width)
    return 1;

  if (integer_bits < ref2.integer_bits)
    return -1;
  if (integer_bits > ref2.integer_bits)
    return 1;

  return 0;
}

code_type2t::code_type2t(void)
  : type_body<code_type2t>(code_id)
{
}

unsigned int
code_type2t::get_width(void) const
{
  assert(0 && "Fetching width of code type - invalid operation");
}

bool
code_type2t::cmp(const code_type2t &ref __attribute__((unused))) const
{
  return true; // All code is the same. Ish.
}

int
code_type2t::lt(const type2t &ref __attribute__((unused))) const
{
  return 0; // All code is the same. Ish.
}

list_of_memberst
code_type2t::tostring(void) const
{
  return list_of_memberst();
}

string_type2t::string_type2t()
  : type_body<string_type2t>(string_id)
{
}

unsigned int
string_type2t::get_width(void) const
{
  assert(0 && "Fetching width of string type - needs consideration");
}

bool
string_type2t::cmp(const string_type2t &ref __attribute__((unused))) const
{
  return true; // All strings are the same.
}

int
string_type2t::lt(const type2t &ref __attribute__((unused))) const
{
  return 0; // All strings are the same.
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

/***************************** Templated expr body ****************************/

template <class derived>
expr_body<derived>::expr_body(const expr_body<derived> &ref)
  : expr2t(ref)
{
}

template <class derived>
void
expr_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc
expr_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
const_expr_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc const_expr_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
const_datatype_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc const_datatype_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
rel_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc rel_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
lops2_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc lops2_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
logic_2ops2t_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc logic_2ops2t_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
binops_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc binops_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
arith_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc arith_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
arith_2ops_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc arith_2ops_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
byte_ops_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc byte_ops_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

template <class derived>
void
datatype_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc datatype_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

/**************************** Expression constructors *************************/

symbol2t::symbol2t(const type2tc type, irep_idt _name)
  : expr_body<symbol2t>(type, symbol_id),
    name(_name)
{
}

symbol2t::symbol2t(const symbol2t &ref)
  : expr_body<symbol2t>(ref),
    name(ref.name)
{
}

bool
symbol2t::cmp(const expr2t &ref) const
{
  const symbol2t &ref2 = static_cast<const symbol2t &>(ref);
  if (name == ref2.name)
    return true;
  return false;
}

int
symbol2t::lt(const expr2t &ref) const
{
  const symbol2t &ref2 = static_cast<const symbol2t &>(ref);
  if (name < ref2.name)
    return -1;
  else if (ref2.name < name)
    return 1;
  else
    return 0;
}

constant_int2t::constant_int2t(type2tc type, const BigInt &input)
  : const_expr_body<constant_int2t>(type, constant_int_id), constant_value(input)
{
}

constant_int2t::constant_int2t(const constant_int2t &ref)
  : const_expr_body<constant_int2t>(ref), constant_value(ref.constant_value)
{
}

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
constant_int2t::cmp(const expr2t &ref) const
{
  const constant_int2t &ref2 = static_cast<const constant_int2t &>(ref);
  if (constant_value == ref2.constant_value)
    return true;
  return false;
}

int
constant_int2t::lt(const expr2t &ref) const
{
  const constant_int2t &ref2 = static_cast<const constant_int2t &>(ref);
  return constant_value.compare(ref2.constant_value);
}

constant_bool2t::constant_bool2t(bool value)
  : const_expr_body<constant_bool2t>(type2tc(new bool_type2t()),
                                     constant_bool_id),
                                     constant_value(value)
{
}

constant_bool2t::constant_bool2t(const constant_bool2t &ref)
  : const_expr_body<constant_bool2t>(ref), constant_value(ref.constant_value)
{
}

bool
constant_bool2t::cmp(const expr2t &ref) const
{
  const constant_bool2t &ref2 = static_cast<const constant_bool2t &>(ref);
  if (constant_value == ref2.constant_value)
    return true;
  return false;
}

int
constant_bool2t::lt(const expr2t &ref) const
{
  const constant_bool2t &ref2 = static_cast<const constant_bool2t &>(ref);
  if (constant_value < ref2.constant_value)
    return -1;
  else if (ref2.constant_value < constant_value)
    return 1;
  else
    return 0;
}

typecast2t::typecast2t(const type2tc type, const expr2tc expr)
  : expr_body<typecast2t>(type, typecast_id), from(expr)
{
}

typecast2t::typecast2t(const typecast2t &ref)
  : expr_body<typecast2t>::expr_body(ref), from(ref.from)
{
}

constant_datatype2t::constant_datatype2t(const type2tc type, expr_ids id,
                                         const std::vector<expr2tc> &members)
  : const_expr_body<constant_datatype2t>(type, id), datatype_members(members)
{
}

constant_datatype2t::constant_datatype2t(const constant_datatype2t &ref)
  : const_expr_body<constant_datatype2t>(ref)
{
}

bool
constant_datatype2t::cmp(const expr2t &ref) const
{
  const constant_datatype2t &ref2 = static_cast<const constant_datatype2t &>
                                               (ref);
  if (datatype_members == ref2.datatype_members)
    return true;
  return false;
}

int
constant_datatype2t::lt(const expr2t &ref) const
{
  const constant_datatype2t &ref2 = static_cast<const constant_datatype2t &>
                                               (ref);
  if (datatype_members < ref2.datatype_members)
    return -1;
  else if (ref2.datatype_members < datatype_members)
    return 1;
  else
    return 0;
}

constant_struct2t::constant_struct2t(const type2tc type,
                                     const std::vector<expr2tc> &members)
  : const_datatype_body<constant_struct2t>(type, constant_struct_id, members)
{
}

constant_struct2t::constant_struct2t(const constant_struct2t &ref)
  : const_datatype_body<constant_struct2t>(ref)
{
}

constant_union2t::constant_union2t(const type2tc type,
                                   const std::vector<expr2tc> &members)
  : const_datatype_body<constant_union2t>(type, constant_union_id, members)
{
}

constant_union2t::constant_union2t(const constant_union2t &ref)
  : const_datatype_body<constant_union2t>(ref)
{
}

constant_string2t::constant_string2t(const type2tc type,
                                     const std::string &stringref)
  : const_expr_body<constant_string2t>(type, constant_string_id), value(stringref)
{
}

constant_string2t::constant_string2t(const constant_string2t &ref)
  : const_expr_body<constant_string2t>(ref), value(ref.value)
{
}

expr2tc
constant_string2t::to_array(void) const
{
  std::vector<expr2tc> contents;
  unsigned int length = value.size(), i;

  unsignedbv_type2t *type = new unsignedbv_type2t(8);
  type2tc tp(type);

  for (i = 0; i < length; i++) {
    constant_int2t *v = new constant_int2t(tp, BigInt(value[i]));
    expr2tc ptr(v);
    contents.push_back(ptr);
  }

  unsignedbv_type2t *len_type = new unsignedbv_type2t(config.ansi_c.int_width);
  type2tc len_tp(len_type);
  constant_int2t *len_val = new constant_int2t(len_tp, BigInt(length));
  expr2tc len_val_ref(len_val);

  array_type2t *arr_type = new array_type2t(tp, len_val_ref, false);
  type2tc arr_tp(arr_type);
  constant_array2t *a = new constant_array2t(arr_tp, contents);

  expr2tc final_val(a);
  return final_val;
}

bool
constant_string2t::cmp(const expr2t &ref) const
{
  const constant_string2t &ref2 = static_cast<const constant_string2t &> (ref);
  if (value == ref2.value)
    return true;
  return false;
}

int
constant_string2t::lt(const expr2t &ref) const
{
  const constant_string2t &ref2 = static_cast<const constant_string2t &> (ref);
  if (value < ref2.value)
    return -1;
  else if (ref2.value < value)
    return 1;
  else
    return 0;
}

constant_array2t::constant_array2t(const type2tc type,
                                   const std::vector<expr2tc> &members)
  : const_expr_body<constant_array2t>(type, constant_array_id),
    datatype_members(members)
{
}

constant_array2t::constant_array2t(const constant_array2t &ref)
  : const_expr_body<constant_array2t>(ref),
    datatype_members(ref.datatype_members)
{
}

bool
constant_array2t::cmp(const expr2t &ref) const
{
  const constant_array2t &ref2 = static_cast<const constant_array2t &> (ref);
  if (datatype_members == ref2.datatype_members)
    return true;
  return false;
}

int
constant_array2t::lt(const expr2t &ref) const
{
  const constant_array2t &ref2 = static_cast<const constant_array2t &> (ref);
  if (datatype_members < ref2.datatype_members)
    return -1;
  else if (ref2.datatype_members < datatype_members)
    return 1;
  else
    return 0;
}

constant_array_of2t::constant_array_of2t(const type2tc type, expr2tc init)
  : const_expr_body<constant_array_of2t>(type, constant_array_id),
    initializer(init)
{
}

constant_array_of2t::constant_array_of2t(const constant_array_of2t &ref)
  : const_expr_body<constant_array_of2t>(ref),
    initializer(ref.initializer)
{
}

bool
constant_array_of2t::cmp(const expr2t &ref) const
{
  const constant_array_of2t &ref2 = static_cast<const constant_array_of2t &>
                                               (ref);
  if (initializer == ref2.initializer)
    return true;
  return false;
}

int
constant_array_of2t::lt(const expr2t &ref) const
{
  const constant_array_of2t &ref2 = static_cast<const constant_array_of2t &>
                                               (ref);
  return initializer->ltchecked(*ref2.initializer.get());
}

if2t::if2t(const type2tc type, const expr2tc _cond, const expr2tc true_val,
           const expr2tc false_val)
  : expr_body<if2t>(type, if_id), cond(_cond), true_value(true_val),
    false_value(false_val)
{
}

if2t::if2t(const if2t &ref) : expr_body<if2t>(ref), cond(ref.cond),
                              true_value(ref.true_value),
                              false_value(ref.false_value)
{
}

bool
if2t::cmp(const expr2t &ref) const
{
  const if2t &ref2 = static_cast<const if2t &>(ref);

  if (cond != ref2.cond)
    return false;

  if (true_value != ref2.true_value)
    return false;

  if (false_value != ref2.false_value)
    return false;

  return true;
}

int
if2t::lt(const expr2t &ref) const
{
  const if2t &ref2 = static_cast<const if2t &>(ref);

  int tmp = cond->ltchecked(*ref2.cond.get());
  if (tmp != 0)
    return tmp;

  tmp = true_value->ltchecked(*ref2.true_value.get());
  if (tmp != 0)
    return tmp;

  return false_value->ltchecked(*ref2.false_value.get());
}

rel2t::rel2t(expr_ids id, const expr2tc val1, const expr2tc val2)
  : expr_body<rel2t>(type2tc(new bool_type2t()), id), side_1(val1), side_2(val2)
{
}

rel2t::rel2t(const rel2t &ref)
  : expr_body<rel2t>(ref)
{
}

bool
rel2t::cmp(const expr2t &ref) const
{
  const rel2t &ref2 = static_cast<const rel2t &>(ref);

  if (side_1 != ref2.side_1)
    return false;

  if (side_2 != ref2.side_2)
    return false;

  return true;
}

int
rel2t::lt(const expr2t &ref) const
{
  const rel2t &ref2 = static_cast<const rel2t &>(ref);

  int tmp = side_1->ltchecked(*ref2.side_1.get());
  if (tmp != 0)
    return tmp;

  return side_2->ltchecked(*ref2.side_2.get());
}

equality2t::equality2t(const expr2tc val1, const expr2tc val2)
  : rel_body<equality2t>(equality_id, val1, val2)
{
}

equality2t::equality2t(const equality2t &ref)
  : rel_body<equality2t>(ref)
{
}

notequal2t::notequal2t(const expr2tc val1, const expr2tc val2)
  : rel_body<notequal2t>(notequal_id, val1, val2)
{
}

notequal2t::notequal2t(const notequal2t &ref)
  : rel_body<notequal2t>(ref)
{
}

lessthan2t::lessthan2t(const expr2tc val1, const expr2tc val2)
  : rel_body<lessthan2t>(lessthan_id, val1, val2)
{
}

lessthan2t::lessthan2t(const lessthan2t &ref)
  : rel_body<lessthan2t>(ref)
{
}

greaterthan2t::greaterthan2t(const expr2tc val1, const expr2tc val2)
  : rel_body<greaterthan2t>(greaterthan_id, val1, val2)
{
}

greaterthan2t::greaterthan2t(const greaterthan2t &ref)
  : rel_body<greaterthan2t>(ref)
{
}

lessthanequal2t::lessthanequal2t(const expr2tc val1, const expr2tc val2)
  : rel_body<lessthanequal2t>(lessthanequal_id, val1, val2)
{
}

lessthanequal2t::lessthanequal2t(const lessthanequal2t &ref)
  : rel_body<lessthanequal2t>(ref)
{
}

greaterthanequal2t::greaterthanequal2t(const expr2tc val1, const expr2tc val2)
  : rel_body<greaterthanequal2t>(greaterthanequal_id, val1, val2)
{
}

greaterthanequal2t::greaterthanequal2t(const greaterthanequal2t &ref)
  : rel_body<greaterthanequal2t>(ref)
{
}

lops2t::lops2t(expr_ids id)
  : expr_body<lops2t>(type2tc(new bool_type2t()), id)
{
}

lops2t::lops2t(const lops2t &ref)
  : expr_body<lops2t>(ref)
{
}

not2t::not2t(const expr2tc val)
  : lops2_body<not2t>(not_id), notvalue(val)
{
}

not2t::not2t(const not2t &ref)
  : lops2_body<not2t>(ref)
{
}

bool
not2t::cmp(const expr2t &ref) const
{
  const not2t &ref2 = static_cast<const not2t &>(ref);
  return notvalue == ref2.notvalue;
}

int
not2t::lt(const expr2t &ref) const
{
  const not2t &ref2 = static_cast<const not2t &>(ref);
  return notvalue->ltchecked(*ref2.notvalue.get());
}

logical_2ops2t::logical_2ops2t(expr_ids id, const expr2tc val1,
                               const expr2tc val2)
  : lops2_body<logical_2ops2t>(id),
    side_1(val1), side_2(val2)
{
}

logical_2ops2t::logical_2ops2t(const logical_2ops2t &ref)
  : lops2_body<logical_2ops2t>(ref), side_1(ref.side_1), side_2(ref.side_2)
{
}

bool
logical_2ops2t::cmp(const expr2t &ref) const
{
  const logical_2ops2t &ref2 = static_cast<const logical_2ops2t &>(ref);

  if (side_1 != ref2.side_1)
    return false;

  if (side_2 != ref2.side_2)
    return false;

  return true;
}

int
logical_2ops2t::lt(const expr2t &ref) const
{
  const logical_2ops2t &ref2 = static_cast<const logical_2ops2t &>(ref);

  int tmp = side_1->ltchecked(*ref2.side_1.get());
  if (tmp != 0)
    return tmp;

  return side_2->ltchecked(*ref2.side_2.get());
}

and2t::and2t(const expr2tc val1, const expr2tc val2)
  : logic_2ops2t_body<and2t>(and_id, val1, val2)
{
}

and2t::and2t(const and2t &ref)
  : logic_2ops2t_body<and2t>(ref)
{
}

or2t::or2t(const expr2tc val1, const expr2tc val2)
  : logic_2ops2t_body<or2t>(or_id, val1, val2)
{
}

or2t::or2t(const or2t &ref)
  : logic_2ops2t_body<or2t>(ref)
{
}

xor2t::xor2t(const expr2tc val1, const expr2tc val2)
  : logic_2ops2t_body<xor2t>(xor_id, val1, val2)
{
}

xor2t::xor2t(const xor2t &ref)
  : logic_2ops2t_body<xor2t>(ref)
{
}

binops2t::binops2t(const type2tc type, expr_ids id,
                   const expr2tc val1, const expr2tc val2)
  : expr_body<binops2t>(type, id), side_1(val1), side_2(val2)
{
}

binops2t::binops2t(const binops2t &ref)
  : expr_body<binops2t>(ref)
{
}

bool
binops2t::cmp(const expr2t &ref) const
{
  const binops2t &ref2 = static_cast<const binops2t &>(ref);

  if (side_1 != ref2.side_1)
    return false;

  if (side_2 != ref2.side_2)
    return false;

  return true;
}

int
binops2t::lt(const expr2t &ref) const
{
  const binops2t &ref2 = static_cast<const binops2t &>(ref);

  int tmp = side_1->ltchecked(*ref2.side_1.get());
  if (tmp != 0)
    return tmp;

  return side_2->ltchecked(*ref2.side_2.get());
}

bitand2t::bitand2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<bitand2t>(type, bitand_id, val1, val2)
{
}

bitand2t::bitand2t(const bitand2t &ref)
  : binops_body<bitand2t>(ref)
{
}

bitor2t::bitor2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<bitor2t>(type, bitor_id, val1, val2)
{
}

bitor2t::bitor2t(const bitor2t &ref)
  : binops_body<bitor2t>(ref)
{
}

bitxor2t::bitxor2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<bitxor2t>(type, bitxor_id, val1, val2)
{
}

bitxor2t::bitxor2t(const bitxor2t &ref)
  : binops_body<bitxor2t>(ref)
{
}

bitnand2t::bitnand2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<bitnand2t>(type, bitnand_id, val1, val2)
{
}

bitnand2t::bitnand2t(const bitnand2t &ref)
  : binops_body<bitnand2t>(ref)
{
}

bitnor2t::bitnor2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<bitnor2t>(type, bitnor_id, val1, val2)
{
}

bitnor2t::bitnor2t(const bitnor2t &ref)
  : binops_body<bitnor2t>(ref)
{
}

bitnxor2t::bitnxor2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<bitnxor2t>(type, bitnxor_id, val1, val2)
{
}

bitnxor2t::bitnxor2t(const bitnxor2t &ref)
  : binops_body<bitnxor2t>(ref)
{
}

lshr2t::lshr2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : binops_body<lshr2t>(type, lshr_id, val1, val2)
{
}

lshr2t::lshr2t(const lshr2t &ref)
  : binops_body<lshr2t>(ref)
{
}

arith2t::arith2t(const type2tc type, expr_ids id)
  : expr_body<arith2t>(type, id)
{
}

arith2t::arith2t(const arith2t &ref)
  : expr_body<arith2t>(ref)
{
}

neg2t::neg2t(const type2tc type, const expr2tc _value)
  : arith_body<neg2t>(type, neg_id), value(_value)
{
}

neg2t::neg2t(const neg2t &ref)
  : arith_body<neg2t>(ref)
{
}

bool
neg2t::cmp(const expr2t &ref) const
{
  const neg2t &ref2 = static_cast<const neg2t &>(ref);
  return value == ref2.value;
}

int
neg2t::lt(const expr2t &ref) const
{
  const neg2t &ref2 = static_cast<const neg2t &>(ref);
  return value->ltchecked(*ref2.value.get());
}

abs2t::abs2t(const type2tc type, const expr2tc _value)
  : arith_body<abs2t>(type, abs_id), value(_value)
{
}

abs2t::abs2t(const abs2t &ref)
  : arith_body<abs2t>(ref)
{
}

bool
abs2t::cmp(const expr2t &ref) const
{
  const abs2t &ref2 = static_cast<const abs2t &>(ref);
  return value == ref2.value;
}

int
abs2t::lt(const expr2t &ref) const
{
  const abs2t &ref2 = static_cast<const abs2t &>(ref);
  return value->ltchecked(*ref2.value.get());
}

arith_2op2t::arith_2op2t(const type2tc type, expr_ids id,
                         const expr2tc val1, const expr2tc val2)
  : arith_body<arith_2op2t>(type, id), part_1(val1), part_2(val2)
{
}

arith_2op2t::arith_2op2t(const arith_2op2t &ref)
  : arith_body<arith_2op2t>(ref)
{
}

bool
arith_2op2t::cmp(const expr2t &ref) const
{
  const arith_2op2t &ref2 = static_cast<const arith_2op2t &>(ref);

  if (part_1 != ref2.part_1)
    return false;

  if (part_2 != ref2.part_2)
    return false;

  return true;
}

int
arith_2op2t::lt(const expr2t &ref) const
{
  const arith_2op2t &ref2 = static_cast<const arith_2op2t &>(ref);

  int tmp = part_1->ltchecked(*ref2.part_1.get());
  if (tmp != 0)
    return tmp;

  return part_2->ltchecked(*ref2.part_2.get());
}

add2t::add2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : arith_2ops_body<add2t>(type, add_id, val1, val2)
{
}

add2t::add2t(const add2t &ref)
  : arith_2ops_body<add2t>(ref)
{
}

sub2t::sub2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : arith_2ops_body<sub2t>(type, sub_id, val1, val2)
{
}

sub2t::sub2t(const sub2t &ref)
  : arith_2ops_body<sub2t>(ref)
{
}

mul2t::mul2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : arith_2ops_body<mul2t>(type, mul_id, val1, val2)
{
}

mul2t::mul2t(const mul2t &ref)
  : arith_2ops_body<mul2t>(ref)
{
}

div2t::div2t(const type2tc type, const expr2tc val1, const expr2tc val2)
  : arith_2ops_body<div2t>(type, div_id, val1, val2)
{
}

div2t::div2t(const div2t &ref)
  : arith_2ops_body<div2t>(ref)
{
}

modulus2t::modulus2t(const type2tc type, const expr2tc val1,const expr2tc val2)
  : arith_2ops_body<modulus2t>(type, modulus_id, val1, val2)
{
}

modulus2t::modulus2t(const modulus2t &ref)
  : arith_2ops_body<modulus2t>(ref)
{
}

shl2t::shl2t(const type2tc type, const expr2tc val1,const expr2tc val2)
  : arith_2ops_body<shl2t>(type, shl_id, val1, val2)
{
}

shl2t::shl2t(const shl2t &ref)
  : arith_2ops_body<shl2t>(ref)
{
}

ashr2t::ashr2t(const type2tc type, const expr2tc val1,const expr2tc val2)
  : arith_2ops_body<ashr2t>(type, ashr_id, val1, val2)
{
}

ashr2t::ashr2t(const ashr2t &ref)
  : arith_2ops_body<ashr2t>(ref)
{
}

same_object2t::same_object2t(const expr2tc val1,const expr2tc val2)
  : arith_2ops_body<same_object2t>(type2tc(new bool_type2t()), same_object_id,
                                   val1, val2)
{
}

same_object2t::same_object2t(const same_object2t &ref)
  : arith_2ops_body<same_object2t>(ref)
{
}

pointer_offset2t::pointer_offset2t(const type2tc type, const expr2tc val)
  : arith_body<pointer_offset2t>(type, pointer_offset_id), pointer_obj(val)
{
}

pointer_offset2t::pointer_offset2t(const pointer_offset2t &ref)
  : arith_body<pointer_offset2t>(ref), pointer_obj(ref.pointer_obj)
{
}

bool
pointer_offset2t::cmp(const expr2t &ref) const
{
  const pointer_offset2t &ref2 = static_cast<const pointer_offset2t &>(ref);
  return pointer_obj == ref2.pointer_obj;
}

int
pointer_offset2t::lt(const expr2t &ref) const
{
  const pointer_offset2t &ref2 = static_cast<const pointer_offset2t &>(ref);
  return pointer_obj->ltchecked(*ref2.pointer_obj.get());
}

pointer_object2t::pointer_object2t(const type2tc type, const expr2tc val)
  : arith_body<pointer_object2t>(type, pointer_object_id), pointer_obj(val)
{
}

pointer_object2t::pointer_object2t(const pointer_object2t &ref)
  : arith_body<pointer_object2t>(ref), pointer_obj(ref.pointer_obj)
{
}

bool
pointer_object2t::cmp(const expr2t &ref) const
{
  const pointer_object2t &ref2 = static_cast<const pointer_object2t &>(ref);
  return pointer_obj == ref2.pointer_obj;
}

int
pointer_object2t::lt(const expr2t &ref) const
{
  const pointer_object2t &ref2 = static_cast<const pointer_object2t &>(ref);
  return pointer_obj->ltchecked(*ref2.pointer_obj.get());
}

address_of2t::address_of2t(const type2tc subtype, const expr2tc val)
  : arith_body<address_of2t>(type2tc(new pointer_type2t(subtype)),
                             address_of_id),
                             pointer_obj(val)
{
}

address_of2t::address_of2t(const address_of2t &ref)
  : arith_body<address_of2t>(ref), pointer_obj(ref.pointer_obj)
{
}

bool
address_of2t::cmp(const expr2t &ref) const
{
  const address_of2t &ref2 = static_cast<const address_of2t &>(ref);
  return pointer_obj == ref2.pointer_obj;
}

int
address_of2t::lt(const expr2t &ref) const
{
  const address_of2t &ref2 = static_cast<const address_of2t &>(ref);
  return pointer_obj->ltchecked(*ref2.pointer_obj.get());
}

byte_ops2t::byte_ops2t(const type2tc type, expr_ids id)
  : expr_body<byte_ops2t>(type, id)
{
}

byte_ops2t::byte_ops2t(const byte_ops2t &ref)
  : expr_body<byte_ops2t>(ref)
{
}

byte_extract2t::byte_extract2t(const type2tc type, bool is_big_endian,
                               const expr2tc source,
                               const expr2tc offs)
  : byte_ops_body<byte_extract2t>(type, byte_extract_id),
                                 big_endian(is_big_endian),
                                 source_value(source),
                                 source_offset(offs)
{
}

byte_extract2t::byte_extract2t(const byte_extract2t &ref)
  : byte_ops_body<byte_extract2t>(ref),
                              big_endian(ref.big_endian),
                              source_value(ref.source_value),
                              source_offset(ref.source_offset)
{
}

bool
byte_extract2t::cmp(const expr2t &ref) const
{
  const byte_extract2t &ref2 = static_cast<const byte_extract2t &>(ref);
 
  if (big_endian != ref2.big_endian)
    return false;

  if (source_value != ref2.source_value)
    return false;

  if (source_offset != ref2.source_offset)
    return false;

  return true;
}

int
byte_extract2t::lt(const expr2t &ref) const
{
  const byte_extract2t &ref2 = static_cast<const byte_extract2t &>(ref);

  if (big_endian < ref2.big_endian)
    return -1;
  if (big_endian > ref2.big_endian)
    return 1;

  int tmp = source_value->ltchecked(*ref2.source_value.get());
  if (tmp != 0)
    return tmp;

  return source_offset->ltchecked(*ref2.source_offset.get());
}

byte_update2t::byte_update2t(const type2tc type, bool is_big_endian,
                             const expr2tc source, const expr2tc offs,
                             const expr2tc update)
  : byte_ops_body<byte_update2t>(type, byte_update_id),
                                 big_endian(is_big_endian),
                                 source_value(source),
                                 source_offset(offs),
                                 update_value(update)
{
}

byte_update2t::byte_update2t(const byte_update2t &ref)
  : byte_ops_body<byte_update2t>(ref),
                              big_endian(ref.big_endian),
                              source_value(ref.source_value),
                              source_offset(ref.source_offset),
                              update_value(ref.update_value)
{
}

bool
byte_update2t::cmp(const expr2t &ref) const
{
  const byte_update2t &ref2 = static_cast<const byte_update2t &>(ref);
 
  if (big_endian != ref2.big_endian)
    return false;

  if (source_value != ref2.source_value)
    return false;

  if (source_offset != ref2.source_offset)
    return false;

  if (update_value != ref2.update_value)
    return false;

  return true;
}

int
byte_update2t::lt(const expr2t &ref) const
{
  const byte_update2t &ref2 = static_cast<const byte_update2t &>(ref);

  if (big_endian < ref2.big_endian)
    return -1;
  if (big_endian > ref2.big_endian)
    return 1;

  int tmp = source_value->ltchecked(*ref2.source_value.get());
  if (tmp != 0)
    return tmp;

  tmp = source_offset->ltchecked(*ref2.source_offset.get());
  if (tmp != 0)
    return tmp;

  return update_value->ltchecked(*ref2.update_value.get());
}

datatype_ops2t::datatype_ops2t(const type2tc type, expr_ids id)
  : expr_body<datatype_ops2t>(type, id)
{
}

datatype_ops2t::datatype_ops2t(const datatype_ops2t &ref)
  : expr_body<datatype_ops2t>(ref)
{
}


with2t::with2t(const type2tc type, const expr2tc source, const expr2tc idx,
               const expr2tc update)
  : datatype_body<with2t>(type, with_id), source_data(source),
                          update_field(idx), update_data(update)
{
}

with2t::with2t(const with2t &ref)
  : datatype_body<with2t>(ref), source_data(ref.source_data),
                         update_field(ref.update_field),
                         update_data(ref.update_data)
{
}

bool
with2t::cmp(const expr2t &ref) const
{
  const with2t &ref2 = static_cast<const with2t &>(ref);
 
  if (source_data != ref2.source_data)
    return false;

  if (update_field != ref2.update_field)
    return false;

  if (update_data != ref2.update_data)
    return false;

  return true;
}

int
with2t::lt(const expr2t &ref) const
{
  const with2t &ref2 = static_cast<const with2t &>(ref);

  int tmp = source_data->ltchecked(*ref2.source_data.get());
  if (tmp != 0)
    return tmp;

  tmp = update_field->ltchecked(*ref2.update_field.get());
  if (tmp != 0)
    return tmp;

  return update_data->ltchecked(*ref2.update_data.get());
}

member2t::member2t(const type2tc type, const expr2tc source,
                   const constant_string2t &idx)
  : datatype_body<member2t>(type, member_id), source_data(source),
                          member(idx)
{
}

member2t::member2t(const member2t &ref)
  : datatype_body<member2t>(ref), source_data(ref.source_data),
                           member(ref.member)
{
}

bool
member2t::cmp(const expr2t &ref) const
{
  const member2t &ref2 = static_cast<const member2t &>(ref);

  if (source_data != ref2.source_data)
    return false;

  if (member != ref2.member)
    return false;

  return true;
}

int
member2t::lt(const expr2t &ref) const
{
  const member2t &ref2 = static_cast<const member2t &>(ref);

  int tmp = source_data->ltchecked(*ref2.source_data.get());
  if (tmp != 0)
    return tmp;

  return member.ltchecked(ref2.member);
}


index2t::index2t(const type2tc type, const expr2tc source,
                 const expr2tc _index)
  : datatype_body<index2t>(type, index_id), source_data(source),
                           index(_index)
{
}

index2t::index2t(const index2t &ref)
  : datatype_body<index2t>(ref), source_data(ref.source_data),
                           index(ref.index)
{
}

bool
index2t::cmp(const expr2t &ref) const
{
  const index2t &ref2 = static_cast<const index2t &>(ref);

  if (source_data != ref2.source_data)
    return false;

  if (index != ref2.index)
    return false;

  return true;
}

int
index2t::lt(const expr2t &ref) const
{
  const index2t &ref2 = static_cast<const index2t &>(ref);

  int tmp = source_data->ltchecked(*ref2.source_data.get());
  if (tmp != 0)
    return tmp;

  return index->ltchecked(*ref2.index.get());
}

zero_string2t::zero_string2t(const expr2tc _string)
  : datatype_body<zero_string2t>(type2tc(new bool_type2t()), zero_string_id),
                                string(_string)
{
}

zero_string2t::zero_string2t(const zero_string2t &ref)
  : datatype_body<zero_string2t>(ref), string(ref.string)
{
}

bool
zero_string2t::cmp(const expr2t &ref) const
{
  const zero_string2t &ref2 = static_cast<const zero_string2t &>(ref);
  return string == ref2.string;
}

int
zero_string2t::lt(const expr2t &ref) const
{
  const zero_string2t &ref2 = static_cast<const zero_string2t &>(ref);
  return string->ltchecked(*ref2.string.get());
}

zero_length_string2t::zero_length_string2t(const expr2tc _string)
  : datatype_body<zero_length_string2t>(type2tc(new bool_type2t()),
                                        zero_length_string_id),
                                        string(_string)
{
}

zero_length_string2t::zero_length_string2t(const zero_length_string2t &ref)
  : datatype_body<zero_length_string2t>(ref), string(ref.string)
{
}

bool
zero_length_string2t::cmp(const expr2t &ref) const
{
  const zero_length_string2t &ref2 = static_cast<const zero_length_string2t &>
                                                (ref);
  return string == ref2.string;
}

int
zero_length_string2t::lt(const expr2t &ref) const
{
  const zero_length_string2t &ref2 = static_cast<const zero_length_string2t &>
                                                (ref);
  return string->ltchecked(*ref2.string.get());
}

isnan2t::isnan2t(const expr2tc val)
  : lops2_body<isnan2t>(is_nan_id), value(val)
{
}

isnan2t::isnan2t(const isnan2t &ref)
  : lops2_body<isnan2t>(ref), value(ref.value)
{
}

bool
isnan2t::cmp(const expr2t &ref) const
{
  const isnan2t &ref2 = static_cast<const isnan2t &> (ref);
  return value == ref2.value;
}

int
isnan2t::lt(const expr2t &ref) const
{
  const isnan2t &ref2 = static_cast<const isnan2t &> (ref);
  return value->ltchecked(*ref2.value.get());
}
