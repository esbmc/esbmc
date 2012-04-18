#include "irep2.h"

#include <solvers/prop/prop_conv.h>

/*************************** Base type2t definitions **************************/

type2t::type2t(type_ids id)
  : type_id(id)
{
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

struct_union_type2t::struct_union_type2t(type_ids id,
                                         const std::vector<type2tc> &_members,
                                         std::vector<std::string> memb_names,
                                         std::string _name)
  : type_body<struct_union_type2t>(id), members(_members),
                                        member_names(memb_names),
                                        name(_name)
{
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

signedbv_type2t::signedbv_type2t(unsigned int width)
  : bv_type_body<signedbv_type2t>(signedbv_id, width)
{
}

unsignedbv_type2t::unsignedbv_type2t(unsigned int width)
  : bv_type_body<unsignedbv_type2t>(unsignedbv_id, width)
{
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

pointer_type2t::pointer_type2t(type2tc _sub)
  : type_body<pointer_type2t>(pointer_id), subtype(_sub)
{
}

unsigned int
pointer_type2t::get_width(void) const
{
  return config.ansi_c.pointer_width;
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

symbol_type2t::symbol_type2t(const dstring sym_name)
  : type_body<symbol_type2t>(symbol_id), symbol_name(sym_name)
{
}

unsigned int
symbol_type2t::get_width(void) const
{
  assert(0 && "Fetching width of symbol type - invalid operation");
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

code_type2t::code_type2t(void)
  : type_body<code_type2t>(code_id)
{
}

unsigned int
code_type2t::get_width(void) const
{
  assert(0 && "Fetching width of code type - invalid operation");
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
