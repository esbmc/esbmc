#include <memory>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/std_types.h>

// Pretty names indexed by expr2t::expr_ids. Driven by expr_kinds.inc;
// adding a new expression kind there automatically populates this
// table and the static_assert below guards against the array drifting
// out of sync with the enum (which is also generated from the .inc).
static const char *expr_names[] = {
#define IREP2_EXPR(kind, pretty) pretty,
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
};
static_assert(
  sizeof(expr_names) == (expr2t::end_expr_id * sizeof(char *)),
  "expr_names[] disagrees with expr2t::expr_ids — somebody edited "
  "the manifest without going through expr_kinds.inc");

void irep2_bad_expr_cast(unsigned actual, unsigned expected, const char *target)
{
  const char *actual_name =
    (actual < expr2t::end_expr_id) ? expr_names[actual] : "<out-of-range>";
  const char *expected_name =
    (expected < expr2t::end_expr_id) ? expr_names[expected] : "<out-of-range>";
  throw irep2_cast_error(fmt::format(
    "irep2: to_{}2t() called on expr whose expr_id is {} (target {})",
    expected_name,
    actual_name,
    target));
}

/*************************** Base expr2t definitions **************************/

expr2t::expr2t(const type2tc &_type, expr_ids id)
  : expr_id(id), type(_type), crc_val(0)
{
}

expr2t::expr2t(const expr2t &ref)
  : irep2t(), expr_id(ref.expr_id), type(ref.type)
{
  // Snapshot the cached CRC. Relaxed is enough: callers must already
  // honour the single-writer contract documented in irep2.h, and the
  // copy is itself a new value not yet visible to anyone else.
  crc_val.store(
    ref.crc_val.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

bool expr2t::operator==(const expr2t &ref) const
{
  if (!expr2t::cmp(ref))
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
  if (tmp < 0)
    return true;
  else if (tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

int expr2t::ltchecked(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if (tmp != 0)
    return tmp;

  return lt(ref);
}

bool expr2t::cmp(const expr2t &ref) const
{
  if (expr_id != ref.expr_id)
    return false;

  if (type != ref.type)
    return false;

  return true;
}

int expr2t::lt(const expr2t &ref) const
{
  if (expr_id < ref.expr_id)
    return -1;
  if (expr_id > ref.expr_id)
    return 1;

  return type->ltchecked(*ref.type.get());
}

size_t expr2t::crc() const
{
  return do_crc();
}

size_t expr2t::do_crc() const
{
  // The atomic crc_val is the cache cell; hash_combine wants a plain
  // size_t reference. Work on a local, then publish.
  size_t v = this->crc_val.load(std::memory_order_relaxed);
  esbmct::hash_combine(v, type->do_crc());
  esbmct::hash_combine(v, (uint8_t)expr_id);
  this->crc_val.store(v, std::memory_order_release);
  return v;
}

void expr2t::hash(crypto_hash &hash) const
{
  static_assert(expr2t::end_expr_id < 256, "Expr id overflow");
  uint8_t eid = expr_id;
  hash.ingest(&eid, sizeof(eid));
  type->hash(hash);
}

std::string get_expr_id(const expr2t &expr)
{
  return std::string(expr_names[expr.expr_id]);
}

std::string expr2t::pretty(unsigned int indent) const
{
  std::string ret =
    pretty_print_func<const expr2t &>(indent, expr_names[expr_id], *this);
  // Dump the type on the end.
  ret += std::string("\n") + indent_str_irep2(indent) +
         "* type : " + type->pretty(indent + 2);
  return ret;
}

void expr2t::dump() const
{
  log_status("{}", pretty(0));
}

template <>
class base_to_names<expr2t>
{
public:
  static constexpr const char **names = expr_names;
};

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

std::string symbol2t::get_symbol_name() const
{
  switch (rlevel)
  {
  case symbol_renaming_level::level0:
    return thename.as_string();
  case symbol_renaming_level::level1:
    return thename.as_string() + "?" + i2string(level1_num) + "!" +
           i2string(thread_num);
  case symbol_renaming_level::level2:
    return thename.as_string() + "?" + i2string(level1_num) + "!" +
           i2string(thread_num) + "&" + i2string(node_num) + "#" +
           i2string(level2_num);
  case symbol_renaming_level::level1_global:
    return thename.as_string();
  case symbol_renaming_level::level2_global:
    return thename.as_string() + "&" + i2string(node_num) + "#" +
           i2string(level2_num);
  }
  assert(0 && "Unrecognized renaming level enum");
  abort();
}

namespace
{
struct constant_string_access
{
  const array_type2t &arr;
  const std::string &s;
  unsigned w; /* element size in bytes */
  bool le;
  size_t n, m; /* array size and value length, in arr.subtype elements */

  explicit constant_string_access(const constant_string2t &e)
    : arr(to_array_type(e.type)),
      s(e.value.as_string()),
      w(arr.subtype->get_width()),
      le(config.ansi_c.endianess == configt::ansi_ct::IS_LITTLE_ENDIAN),
      n(to_constant_int2t(arr.array_size).value.to_uint64())
  {
    assert(config.ansi_c.endianess != configt::ansi_ct::NO_ENDIANESS);
    assert(w % 8 == 0);
    w /= 8;
    assert(0 < w && w <= 4);
    assert(s.length() % w == 0);
    m = s.length() / w;
  }

  constant_string_access(const constant_string_access &) = delete;
  constant_string_access &operator=(const constant_string_access &) = delete;

  expr2tc operator[](size_t i) const
  {
    if (i >= n) /* not in array */
      return expr2tc();

    uint32_t c = 0;
    if (i < m) /* not the '\0' element */
      for (unsigned j = 0; j < w; j++)
        c |= (uint32_t)(unsigned char)s[w * i + j] << 8 * (le ? j : w - 1 - j);
    return gen_long(arr.subtype, c);
  }
};
} // namespace

expr2tc constant_string2t::to_array() const
{
  constant_string_access csa(*this);
  std::vector<expr2tc> contents(csa.n);

  for (size_t i = 0; i < csa.n; i++)
    contents[i] = csa[i];

  expr2tc r = constant_array2tc(type, std::move(contents));
  return r;
}

size_t constant_string2t::array_size() const
{
  return to_constant_int2t(to_array_type(type).array_size).value.to_uint64();
}

expr2tc constant_string2t::at(size_t i) const
{
  constant_string_access csa(*this);
  expr2tc r = csa[i];
  return r;
}

static void assert_type_compat_for_with(const type2tc &a, const type2tc &b)
{
  if (is_array_type(a))
  {
    assert(is_array_type(b));
    const array_type2t &at = to_array_type(a);
    const array_type2t &bt = to_array_type(b);
    assert_type_compat_for_with(at.subtype, bt.subtype);
    assert(at.size_is_infinite == bt.size_is_infinite);
    if (at.size_is_infinite)
      return;
    if (is_symbol2t(at.array_size) || is_symbol2t(bt.array_size))
      return;
    assert(at.array_size == bt.array_size);
  }
  else if (is_code_type(a))
  {
    assert(is_code_type(b));
    const code_type2t &at [[maybe_unused]] = to_code_type(a);
    const code_type2t &bt [[maybe_unused]] = to_code_type(b);
    assert(at.arguments == bt.arguments);
    assert(at.ret_type == bt.ret_type);
    /* don't compare argument names, they could be empty on one side */
    assert(at.ellipsis == bt.ellipsis);
  }
  else if (is_empty_type(a) || is_empty_type(b))
    return;
  else if (is_pointer_type(a))
  {
    assert(is_pointer_type(b));
    assert_type_compat_for_with(
      to_pointer_type(a).subtype, to_pointer_type(b).subtype);
  }
  else
    assert(a == b);
}

void with2t::assert_consistency() const
{
  if (is_array_type(source_value))
  {
    assert(
      is_bv_type(update_field->type) || is_pointer_type(update_field->type));
    const array_type2t &arr_type = to_array_type(source_value->type);
    assert_type_compat_for_with(arr_type.subtype, update_value->type);
  }
  else if (is_vector_type(source_value))
  {
    assert(is_bv_type(update_field->type));
    assert_type_compat_for_with(
      to_vector_type(source_value->type).subtype, update_value->type);
  }
  else
  {
    const struct_union_data *d =
      dynamic_cast<const struct_union_data *>(source_value->type.get());
    assert(d);
    assert(update_field->expr_id == constant_string_id);
    auto c = d->get_component_number(to_constant_string2t(update_field).value);
    assert(c.has_value());
    assert_type_compat_for_with(update_value->type, d->members[*c]);
  }
  assert(type == source_value->type);
}

const expr2tc &object_descriptor2t::get_root_object() const
{
  const expr2tc *tmp = &object;

  do
  {
    if (is_member2t(*tmp))
      tmp = &to_member2t(*tmp).source_value;
    else if (is_index2t(*tmp))
      tmp = &to_index2t(*tmp).source_value;
    else
      return *tmp;
  } while (1);
}

printf_kindt printf_kind_from_name(const irep_idt &name)
{
  if (name == "printf")
    return printf_kindt::PRINTF;
  if (name == "fprintf")
    return printf_kindt::FPRINTF;
  if (name == "dprintf")
    return printf_kindt::DPRINTF;
  if (name == "sprintf")
    return printf_kindt::SPRINTF;
  if (name == "vfprintf")
    return printf_kindt::VFPRINTF;
  if (name == "snprintf")
    return printf_kindt::SNPRINTF;
  assert(0 && "Unrecognized printf-family base_name");
  abort();
}

void assert_arith_2ops_consistency(
  [[maybe_unused]] const type2tc &t,
  [[maybe_unused]] expr2t::expr_ids id,
  [[maybe_unused]] const expr2tc &v1,
  [[maybe_unused]] const expr2tc &v2)
{
#ifndef NDEBUG /* only check consistency in non-Release builds */
  bool p1 = is_pointer_type(v1);
  bool p2 = is_pointer_type(v2);
  auto is_bv_type = [](const type2tc &t) {
    return t->type_id == type2t::unsignedbv_id ||
           t->type_id == type2t::signedbv_id;
  };
  if (p1 && p2)
  {
    assert(id == expr2t::sub_id);
    assert(is_bv_type(t));
    assert(t->get_width() == config.ansi_c.address_width);
  }
  else if (!(is_vector_type(v1->type) || is_vector_type(v2->type)))
  {
    assert(
      p2 || (is_bv_type(t) == is_bv_type(v1->type) &&
             t->get_width() == v1->type->get_width()));
    assert(
      p1 || (is_bv_type(t) == is_bv_type(v2->type) &&
             t->get_width() == v2->type->get_width()));
  }
  // TODO: Add consistency checks for vectors
#endif
}

