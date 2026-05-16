#include <memory>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <irep2/irep2_dispatch.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/std_types.h>

/*************************** Base type2t definitions **************************/

// Pretty names indexed by type2t::type_ids. Driven by type_kinds.inc;
// adding a new type kind there automatically populates this table.
// The static_assert guards against the array drifting out of sync
// with the enum (which is also generated from the same .inc).
static const char *type_names[] = {
#define IREP2_TYPE(kind, pretty) pretty,
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
};
static_assert(
  sizeof(type_names) == (type2t::end_type_id * sizeof(char *)),
  "type_names[] disagrees with type2t::type_ids — somebody edited "
  "the manifest without going through type_kinds.inc");

std::string get_type_id(const type2t &type)
{
  return std::string(type_names[type.type_id]);
}

void irep2_bad_type_cast(unsigned actual, unsigned expected, const char *target)
{
  const char *actual_name =
    (actual < type2t::end_type_id) ? type_names[actual] : "<out-of-range>";
  const char *expected_name =
    (expected < type2t::end_type_id) ? type_names[expected] : "<out-of-range>";
  throw irep2_cast_error(fmt::format(
    "irep2: to_{}_type() called on type whose type_id is {} (target {})",
    expected_name,
    actual_name,
    target));
}

type2t::type2t(type_ids id) : type_id(id), crc_val(0)
{
}

type2t::type2t(const type2t &ref) : irep2t(), type_id(ref.type_id)
{
  // Snapshot the cached CRC under the single-writer contract; see
  // irep2.h header note. The fresh atomic starts with whatever value
  // ref had at this moment, or 0 if ref had not been crc-ed yet.
  crc_val.store(
    ref.crc_val.load(std::memory_order_relaxed), std::memory_order_relaxed);
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
  if (tmp < 0)
    return true;
  else if (tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

int type2t::ltchecked(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if (tmp != 0)
    return tmp;

  return lt(ref);
}

bool type2t::cmpchecked(const type2t &ref) const
{
  if (type_id == ref.type_id)
    return cmp(ref);

  return false;
}

std::string type2t::pretty(unsigned int indent) const
{
  return pretty_print_func<const type2t &>(indent, type_names[type_id], *this);
}

void type2t::dump() const
{
  log_status("{}", pretty(0));
}


unsigned int bool_type2t::get_width() const
{
  // For the purpose of the byte representing memory model
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
  if (size_is_infinite)
    throw inf_sized_array_excp();

  if (array_size->expr_id != expr2t::constant_int_id)
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

unsigned int pointer_type2t::get_width() const
{
  /* CHERI-TODO: take into account whether we can-carry-provenance. */
  return config.ansi_c.pointer_width();
}

unsigned int empty_type2t::get_width() const
{
  throw symbolic_type_excp();
}

unsigned int symbol_type2t::get_width() const
{
  throw symbolic_type_excp();
}

unsigned int cpp_name_type2t::get_width() const
{
  throw symbolic_type_excp();
}

unsigned int struct_type2t::get_width() const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); ++it)
    width += (*it)->get_width();

  return width;
}

unsigned int union_type2t::get_width() const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); ++it)
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

unsigned int complex_type2t::get_width() const
{
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); ++it)
    width += (*it)->get_width();

  return width;
}

unsigned int code_data::get_width() const
{
  throw symbolic_type_excp();
}

/********************** Switch-based v2 dispatchers ***************************/
// Step 1 of issue #4560: each case delegates to the existing virtual method.
// `end_type_id` is a sentinel never used as a live id; suppress the Wswitch
// noise it generates while keeping per-kind exhaustiveness via the X-macro.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"

bool type2t::cmp(const type2t &o) const
{
  if (type_id != o.type_id)
    return false;
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_cmp_type(                                           \
      static_cast<const kind##_type2t &>(*this), o);
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

int type2t::lt(const type2t &o) const
{
  if (type_id != o.type_id)
    return type_id < o.type_id ? -1 : 1;
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_lt_type(                                            \
      static_cast<const kind##_type2t &>(*this), o);
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

type2tc type2t::clone() const
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_clone_type(                                         \
      static_cast<const kind##_type2t &>(*this));
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

size_t type2t::crc() const
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_do_crc_type(                                        \
      static_cast<const kind##_type2t &>(*this));
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

void type2t::hash(crypto_hash &h) const
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    esbmct::generic_hash_type(                                                 \
      static_cast<const kind##_type2t &>(*this), h);                           \
    return;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

list_of_memberst type2t::tostring(unsigned int indent) const
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_tostring_type(                                      \
      static_cast<const kind##_type2t &>(*this), indent);
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

void type2t::foreach_subtype_impl_const(const_subtype_delegate &f) const
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    esbmct::generic_foreach_subtype_const(                                     \
      static_cast<const kind##_type2t &>(*this), f);                           \
    return;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

void type2t::foreach_subtype_impl(subtype_delegate &f)
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    esbmct::generic_foreach_subtype(                                           \
      static_cast<kind##_type2t &>(*this), f);                                 \
    return;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

unsigned int type2t::get_width() const
{
  switch (type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case kind##_id:                                                              \
    return static_cast<const kind##_type2t &>(*this).get_width();
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  }
  __builtin_unreachable();
}

#pragma GCC diagnostic pop

// Field-name tables consumed by generic_tostring_type. Indexed by the
// order in each type kind's `fields` tuple.
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
std::string vector_type2t::field_names[esbmct::num_type_fields] =
  {"subtype", "array_size", "size_is_infinite", "", ""};
std::string pointer_type2t::field_names[esbmct::num_type_fields] =
  {"subtype", "provenance", "", "", ""};
std::string fixedbv_type2t::field_names[esbmct::num_type_fields] =
  {"width", "integer_bits", "", "", ""};
std::string floatbv_type2t::field_names[esbmct::num_type_fields] =
  {"fraction", "exponent", "", "", ""};
std::string complex_type2t::field_names[esbmct::num_type_fields] =
  {"members", "member_names", "member_pretty_names", "typename", "packed", ""};
std::string cpp_name_type2t::field_names[esbmct::num_type_fields] =
  {"name", "template args", "", "", ""};

const std::vector<type2tc> &struct_union_data::get_structure_members() const
{
  return members;
}

const std::vector<irep_idt> &
struct_union_data::get_structure_member_names() const
{
  return member_names;
}

std::optional<unsigned int>
struct_union_data::get_component_number(const irep_idt &comp) const
{
  unsigned int i = 0, count = 0, pos = 0;
  for (auto const &it : member_names)
  {
    if (it == comp)
    {
      pos = i;
      ++count;
    }
    i++;
  }

  if (count == 1)
    return pos;
  return std::nullopt;
}
