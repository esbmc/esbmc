#include "irep2.h"

#include <solvers/prop/prop_conv.h>

/*************************** Base type2t definitions **************************/

type2t::type2t(type_ids id)
  : type_id(id)
{
}

bool
type2t::operator!=(const type2t &ref) const
{

  return !(*this == ref);
}

template<class derived>
void
type_body<derived>::convert_smt_type(const prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

template<class derived>
bool
type_body<derived>::operator==(const type2t &ref) const
{

  if (type_id != ref.type_id)
    return false;

  const derived &castedref = dynamic_cast<const derived&>(ref);
  const derived &thiscasted = dynamic_cast<const derived&>(*this);
  return thiscasted.cmp(castedref);
}

template<class derived>
void
bv_type_body<derived>::convert_smt_type(const prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

template<class derived>
bool
bv_type_body<derived>::operator==(const type2t &ref) const
{

  if (type_id != ref.type_id)
    return false;

  const derived &castedref = dynamic_cast<const derived&>(ref);
  const derived &thiscasted = dynamic_cast<const derived&>(*this);
  return thiscasted.cmp(castedref);
}

template<class derived>
void
struct_union_type_body2t<derived>::convert_smt_type(const prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

template<class derived>
bool
struct_union_type_body2t<derived>::operator==(const type2t &ref) const
{

  if (type_id != ref.type_id)
    return false;

  const derived &castedref = dynamic_cast<const derived&>(ref);
  const derived &thiscasted = dynamic_cast<const derived&>(*this);
  return thiscasted.cmp(castedref);
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

signedbv_type2t::signedbv_type2t(unsigned int width)
  : bv_type_body<signedbv_type2t>(signedbv_id, width)
{
}

bool
signedbv_type2t::cmp(const signedbv_type2t &ref) const
{
  return bv_type2t::cmp(ref);
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
  return (array_size->type == ref.array_size->type);
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

fixedbv_type2t::fixedbv_type2t(unsigned int fraction, unsigned int integer)
  : type_body<fixedbv_type2t>(fixedbv_id), fraction_bits(fraction),
                                           integer_bits(integer)
{
}

unsigned int
fixedbv_type2t::get_width(void) const
{
  return fraction_bits;
}

bool
fixedbv_type2t::cmp(const fixedbv_type2t &ref) const
{

  if (fraction_bits != ref.fraction_bits)
    return false;

  if (integer_bits != ref.integer_bits)
    return false;

  return true;
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
code_type2t::cmp(const code_type2t &ref) const
{
  return true; // All code is the same. Ish.
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
string_type2t::cmp(const string_type2t &ref) const
{
  return true; // All strings are the same.
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

rel2t::rel2t(expr_ids id, const expr2tc val1, const expr2tc val2)
  : expr_body<rel2t>(type2tc(new bool_type2t()), id), side_1(val1), side_2(val2)
{
}

rel2t::rel2t(const rel2t &ref)
  : expr_body<rel2t>(ref)
{
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

abs2t::abs2t(const type2tc type, const expr2tc _value)
  : arith_body<abs2t>(type, abs_id), value(_value)
{
}

abs2t::abs2t(const abs2t &ref)
  : arith_body<abs2t>(ref)
{
}
