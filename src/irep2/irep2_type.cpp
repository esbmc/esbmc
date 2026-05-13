#include <memory>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
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

type2t::type2t(const type2t &ref) : type_id(ref.type_id)
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

int type2t::lt(const type2t &ref) const
{
  if (type_id < ref.type_id)
    return -1;
  if (type_id > ref.type_id)
    return 1;
  return 0;
}

std::string type2t::pretty(unsigned int indent) const
{
  return pretty_print_func<const type2t &>(indent, type_names[type_id], *this);
}

void type2t::dump() const
{
  log_status("{}", pretty(0));
}

size_t type2t::crc() const
{
  return do_crc();
}

size_t type2t::do_crc() const
{
  size_t v = this->crc_val.load(std::memory_order_relaxed);
  esbmct::hash_combine(v, (uint8_t)type_id);
  this->crc_val.store(v, std::memory_order_release);
  return v;
}

void type2t::hash(crypto_hash &hash) const
{
  static_assert(type2t::end_type_id < 256, "Type id overflow");
  uint8_t tid = type_id;
  hash.ingest(&tid, sizeof(tid));
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
