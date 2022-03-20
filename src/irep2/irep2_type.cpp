#include <memory>
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

/*************************** Base type2t definitions **************************/

static const char *type_names[] = {
  "bool",
  "empty",
  "symbol",
  "struct",
  "union",
  "code",
  "array",
  "vector",
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

template <>
class base_to_names<type2t>
{
public:
  static constexpr const char **names = type_names;
};

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
    throw inf_sized_array_excp();

  if(array_size->expr_id != expr2t::constant_int_id)
    throw dyn_sized_array_excp(array_size);

  // Otherwise, we can multiply the size of the subtype by the number of elements.
  unsigned int sub_width = subtype->get_width();

  const expr2t *elem_size = array_size.get();
  const constant_int2t *const_elem_size =
    dynamic_cast<const constant_int2t *>(elem_size);
  assert(const_elem_size != nullptr);
  unsigned long num_elems = const_elem_size->as_ulong();

  return num_elems * sub_width;
}

unsigned int vector_type2t::get_width() const
{
  unsigned int sub_width = subtype->get_width();

  const expr2t *elem_size = array_size.get();
  const constant_int2t *const_elem_size =
    dynamic_cast<const constant_int2t *>(elem_size);
  assert(const_elem_size != nullptr);
  unsigned long num_elems = const_elem_size->as_ulong();

  return num_elems * sub_width;
}

expr2tc vector_type2t::distribute_operation(
  std::function<expr2tc(type2tc, expr2tc, expr2tc)> func,
  expr2tc op1,
  expr2tc op2)
{
  /*
   * If both op1 and op2 are vectors the resulting value
   * would be the operation over each member
   *
   * Example:
   *
   * op1 = {1,2,3,4}
   * op2 = {1,1,1,1}
   * func = add
   *
   * This would result in:
   *
   * { add(op1[0], op2[0]), add(op1[1], op2[1]), ...}
   * {2,3,4,5}
   */
  if(is_constant_vector2t(op1) && is_constant_vector2t(op2))
  {
    constant_vector2tc vec1(op1);
    constant_vector2tc vec2(op2);
    for(size_t i = 0; i < vec1->datatype_members.size(); i++)
    {
      auto &A = vec1->datatype_members[i];
      auto &B = vec2->datatype_members[i];
      auto new_op = func(A->type, A, B);
      vec1->datatype_members[i] = new_op;
    }
    return vec1;
  }
  /*
   * If only one of the operator is a vector, then the result
   * would extract each value of the vector and apply the value to
   * the other operator
   *
   * Example:
   *
   * op1 = {1,2,3,4}
   * op2 = 1
   * func = add
   *
   * This would result in:
   *
   * { add(op1[0], 1), add(op1[1], 1), ...}
   * {2,3,4,5}
   */
  else
  {
    bool is_op1_vec = is_constant_vector2t(op1);
    expr2tc c = !is_op1_vec ? op1 : op2;
    constant_vector2tc vector(is_op1_vec ? op1 : op2);
    for(auto &datatype_member : vector->datatype_members)
    {
      auto &op = datatype_member;
      auto e1 = is_op1_vec ? op : c;
      auto e2 = is_op1_vec ? c : op;
      auto new_op = func(op->type, e1, e2);
      datatype_member = new_op->do_simplify();
    }
    return vector;
  }
}

expr2tc vector_type2t::distribute_operation(
  expr2t::expr_ids id,
  expr2tc op1,
  expr2tc op2,
  expr2tc rm)
{
#ifndef NDEBUG
  assert(is_vector_type(op1) || (op2 && is_vector_type(op2)));
#endif
  auto is_op1_vector = is_vector_type(op1);
  auto vector_length = is_op1_vector ? to_vector_type(op1->type).array_size
                                     : to_vector_type(op2->type).array_size;
  assert(is_constant_int2t(vector_length));

  auto vector_subtype = is_op1_vector ? to_vector_type(op1->type).subtype
                                      : to_vector_type(op2->type).subtype;
  auto result = is_op1_vector ? gen_zero(op1->type) : gen_zero(op2->type);

  /*
   * If both op1 and op2 are vectors the resulting value
   * would be the operation over each member
   *
   * Example:
   *
   * op1 = {1,2,3,4}
   * op2 = {1,1,1,1}
   * func = add
   *
   * This would result in:
   *
   * { add(op1[0], op2[0]), add(op1[1], op2[1]), ...}
   * {2,3,4,5}
   */

  /*
   * If only one of the operator is a vector, then the result
   * would extract each value of the vector and apply the value to
   * the other operator
   *
   * Example:
   *
   * op1 = {1,2,3,4}
   * op2 = 1
   * func = add
   *
   * This would result in:
   *
   * { add(op1[0], 1), add(op1[1], 1), ...}
   * {2,3,4,5}
   */

  auto is_op_between_vectors =
    is_vector_type(op1) && (op2 && is_vector_type(op2));
  for(size_t i = 0; i < to_constant_int2t(vector_length).as_ulong(); i++)
  {
    BigInt position(i);

    expr2tc local_op1 = op1;
    if(is_vector_type(op1->type))
    {
      local_op1 = index2tc(
        to_vector_type(op1->type).subtype,
        op1,
        constant_int2tc(get_uint32_type(), position));
    }

    expr2tc local_op2 = op2;
    if(op2 && is_vector_type(op2->type))
    {
      local_op2 = index2tc(
        to_vector_type(op1->type).subtype,
        op2,
        constant_int2tc(get_uint32_type(), position));
    }

    expr2tc to_add;
    switch(id)
    {
    case expr2t::neg_id:
      to_add = neg2tc(to_vector_type(op1->type).subtype, local_op1);
      break;
    case expr2t::bitnot_id:
      to_add = bitnot2tc(to_vector_type(op1->type).subtype, local_op1);
      break;
    case expr2t::sub_id:
      to_add = sub2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::mul_id:
      to_add = mul2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::div_id:
      to_add = div2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::modulus_id:
      to_add =
        modulus2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::add_id:
      to_add = add2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::shl_id:
      to_add = shl2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::bitxor_id:
      to_add =
        bitxor2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::bitor_id:
      to_add =
        bitor2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::bitand_id:
      to_add =
        bitand2tc(to_vector_type(op1->type).subtype, local_op1, local_op2);
      break;
    case expr2t::ieee_add_id:
      to_add = ieee_add2tc(
        to_vector_type(op1->type).subtype, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_div_id:
      to_add = ieee_div2tc(
        to_vector_type(op1->type).subtype, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_sub_id:
      to_add = ieee_sub2tc(
        to_vector_type(op1->type).subtype, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_mul_id:
      to_add = ieee_mul2tc(
        to_vector_type(op1->type).subtype, local_op1, local_op2, rm);
      break;
    default:
      assert(0 && "Unsupported operation for Vector");
      abort();
    }
    to_constant_vector2t(result).datatype_members[i] = to_add;
  }
  return result;
}

unsigned int pointer_type2t::get_width() const
{
  return config.ansi_c.pointer_width;
}

unsigned int empty_type2t::get_width() const
{
  throw symbolic_type_excp();
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
  throw symbolic_type_excp();
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
