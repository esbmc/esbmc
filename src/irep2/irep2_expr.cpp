#include <memory>
#include <charconv>
#include <unordered_map>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <irep2/irep2_dispatch.h>
#include <util/message/format.h>
#include <util/migrate.h>

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

// For CRCing to actually be accurate, expr ids mustn't overflow out of a
// byte. If this happens then a) there are too many exprs, and b) the
// expr crcing code has to change.
static_assert(expr2t::end_expr_id <= 256, "Expr id overflow");

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
  return cmp(ref);
}

bool expr2t::operator!=(const expr2t &ref) const
{
  return !(*this == ref);
}

bool expr2t::operator<(const expr2t &ref) const
{
  return lt(ref) < 0;
}

std::string get_expr_id(const expr2t &expr)
{
  return std::string(expr_names[expr.expr_id]);
}

std::string expr2t::pretty(unsigned int indent) const
{
  list_of_memberst memb = tostring(indent + 2);
  std::string indentstr = indent_str_irep2(indent);
  std::string ret = expr_names[expr_id];
  for (auto const &m : memb)
    ret += "\n" + indentstr + "* " + m.first + " : " + m.second;
  ret += "\n" + indentstr + "* type : " + type->pretty(indent + 2);
  return ret;
}

void expr2t::dump() const
{
  log_status("{}", pretty(0));
}

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
  // The fully-qualified SSA name is a pure function of the symbol's
  // (thename, rlevel, l1, thread, node, l2) identity fields, which are
  // immutable for a given node. symex requests the same symbol's name many
  // times (~18x measured on test_locks_13), and each rebuild allocated
  // several short-lived strings. Memoise via a thread-local side table keyed
  // by that identity: the qualified name accounted for ~16% of runtime
  // (mostly i2string + string concatenation), and caching removes the bulk
  // of it.
  //
  // The cache lives outside the node because irep2's node-layout invariant
  // (fields_cover_class) does not admit a non-identity member on the node.
  // thread_local keeps it correct under the single-writer-per-thread
  // contract without locking; values are interned irep_idt (4 bytes) so the
  // table stays compact regardless of name length.
  struct keyt
  {
    unsigned name_no, l1, thr, node, l2;
    int lvl;
    bool operator==(const keyt &o) const
    {
      return name_no == o.name_no && l1 == o.l1 && thr == o.thr &&
             node == o.node && l2 == o.l2 && lvl == o.lvl;
    }
  };
  struct key_hash
  {
    std::size_t operator()(const keyt &k) const
    {
      std::size_t h = k.name_no;
      h = h * 1000003u + k.l1;
      h = h * 1000003u + k.thr;
      h = h * 1000003u + k.node;
      h = h * 1000003u + k.l2;
      h = h * 1000003u + (unsigned)k.lvl;
      return h;
    }
  };
  static thread_local std::unordered_map<keyt, irep_idt, key_hash> memo;

  keyt key{
    thename.get_no(),
    level1_num,
    thread_num,
    node_num,
    level2_num,
    (int)rlevel};
  auto it = memo.find(key);
  if (it != memo.end())
    return it->second.as_string();

  // Build the qualified name into a single pre-sized buffer. The previous
  // `as_string() + "?" + i2string(n) + ...` form allocated ~9 temporary
  // strings per name (each i2string is an sprintf + heap string, and each
  // operator+ reallocates the growing prefix). Appending decimal digits in
  // place with std::to_chars keeps it to one allocation and no sprintf.
  auto append_uint = [](std::string &out, unsigned v) {
    char buf[10]; // unsigned <= 4294967295 -> at most 10 digits
    auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    (void)ec;
    out.append(buf, end);
  };

  const std::string &base = thename.as_string();
  std::string built;
  built.reserve(base.size() + 32);
  built = base;
  switch (rlevel)
  {
  case symbol_renaming_level::level0:
  case symbol_renaming_level::level1_global:
    break;
  case symbol_renaming_level::level1:
    built += '?';
    append_uint(built, level1_num);
    built += '!';
    append_uint(built, thread_num);
    break;
  case symbol_renaming_level::level2:
    built += '?';
    append_uint(built, level1_num);
    built += '!';
    append_uint(built, thread_num);
    built += '&';
    append_uint(built, node_num);
    built += '#';
    append_uint(built, level2_num);
    break;
  case symbol_renaming_level::level2_global:
    built += '&';
    append_uint(built, node_num);
    built += '#';
    append_uint(built, level2_num);
    break;
  default:
    assert(0 && "Unrecognized renaming level enum");
    abort();
  }
  memo.emplace(key, irep_idt(built));
  return built;
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
    assert(
      is_structure_type(source_value->type) ||
      is_complex_type(source_value->type));
    assert(update_field->expr_id == constant_string_id);
    auto c = struct_union_get_component_number(
      source_value->type, to_constant_string2t(update_field).value);
    assert(c.has_value());
    assert_type_compat_for_with(
      update_value->type, struct_union_members(source_value->type)[*c]);
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

/********************** Switch-based v2 dispatchers ***************************/
// All 111 expr kinds now expose `fields`; every case uses the generic path.
// `end_expr_id` is a sentinel never assigned to a live node; including it as
// a switch case (with -Wswitch enabled) makes the compiler enforce per-kind
// exhaustiveness via the X-macro — adding a new kind without wiring it into
// expr_kinds.inc fails to compile here.

bool expr2t::cmp(const expr2t &o) const
{
  if (expr_id != o.expr_id)
    return false;
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_cmp(static_cast<const kind##2t &>(*this), o);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

int expr2t::lt(const expr2t &o) const
{
  if (expr_id != o.expr_id)
    return expr_id < o.expr_id ? -1 : 1;
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_lt(static_cast<const kind##2t &>(*this), o);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

expr2tc expr2t::clone() const
{
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return make_irep<kind##2t>(static_cast<const kind##2t &>(*this));
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

namespace
{
// Compile-time trait: K supports type substitution iff
//   * K::fields is non-empty AND
//   * the first member pointer is a `(const) type2tc expr2t::*` (i.e. it
//     refers to expr2t::type) AND
//   * K is constructible from `(const type2tc &, rest...)` where rest...
//     is the value types of the remaining fields.
//
// The first-field check rules out kinds that derive their type from
// operands and list a different first field (constant_bool, relation
// ops, ...). The constructor check rules out kinds whose `fields` tuple
// starts with `&expr2t::type` for cmp/crc/hash purposes but whose
// constructor synthesises the type from operands and rejects a leading
// `type2tc` (e.g. and/or/xor/implies/isinstance/hasattr/isnone,
// signbit/popcount, code_cpp_throw_decl[_end]). Treating those as
// supported would silently fail to instantiate at the make_irep call
// inside `rebuild_with_type_impl`; the trait pushes the failure to a
// readable "kind unsupported" path instead.
template <class K, std::size_t... Is>
constexpr bool ctor_takes_type_first(std::index_sequence<Is...>)
{
  using fields_t = std::remove_cv_t<decltype(K::fields)>;
  return std::is_constructible_v<
    K,
    const type2tc &,
    const typename esbmct::member_traits<std::remove_cvref_t<
      std::tuple_element_t<Is + 1, fields_t>>>::member_t &...>;
}

template <class K>
constexpr bool supports_with_type_v = []() {
  using fields_t = std::remove_cv_t<decltype(K::fields)>;
  if constexpr (std::tuple_size_v<fields_t> == 0)
    return false;
  else
  {
    using first_t = std::remove_cvref_t<std::tuple_element_t<0, fields_t>>;
    using first_class_t = typename esbmct::member_traits<first_t>::class_t;
    using first_member_t = typename esbmct::member_traits<first_t>::member_t;
    if constexpr (
      !std::is_same_v<first_class_t, expr2t> ||
      !std::is_same_v<first_member_t, type2tc>)
      return false;
    else
      return ctor_takes_type_first<K>(
        std::make_index_sequence<std::tuple_size_v<fields_t> - 1>{});
  }
}();

// rebuild_with_type<K> is only instantiated for kinds where
// supports_with_type_v<K> holds. The dispatcher gates the call on that
// trait; instantiating for an unsupported kind is a programmer error
// that the static_assert below catches at compile time.
template <class K, std::size_t... Is>
expr2tc rebuild_with_type_impl(
  const K &k,
  const type2tc &new_type,
  std::index_sequence<Is...>)
{
  static_assert(
    supports_with_type_v<K>,
    "with_type called on a kind whose first field is not &expr2t::type. "
    "This kind derives its type from operands (e.g. constant_bool, "
    "relation ops) and cannot have its type substituted. Either teach "
    "the kind to list &expr2t::type first and accept it in its primary "
    "constructor, or restructure the caller so it does not need "
    "with_type for this kind.");
  // First field is the type slot; rebuild from new_type + the rest.
  return make_irep<K>(new_type, (k.*std::get<Is + 1>(K::fields))...);
}

template <class K>
expr2tc rebuild_with_type(const K &k, const type2tc &new_type)
{
  constexpr std::size_t N = std::tuple_size_v<decltype(K::fields)>;
  return rebuild_with_type_impl(k, new_type, std::make_index_sequence<N - 1>{});
}

[[noreturn]] void with_type_unsupported(const expr2t &e)
{
  log_error(
    "with_type called on kind {} which has no substitutable type",
    get_expr_id(e));
  abort();
}
} // namespace

expr2tc expr2t::with_type(const type2tc &new_type) const
{
  // Dispatcher: instantiate rebuild_with_type<K> only for kinds that
  // support it. Unsupported kinds go to a shared runtime error path; the
  // static_assert inside rebuild_with_type then guards against accidentally
  // instantiating it for an unsupported K from anywhere else.
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    if constexpr (supports_with_type_v<kind##2t>)                              \
      return rebuild_with_type(                                                \
        static_cast<const kind##2t &>(*this), new_type);                       \
    else                                                                       \
      with_type_unsupported(*this);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

list_of_memberst expr2t::tostring(unsigned int indent) const
{
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_tostring(                                           \
      static_cast<const kind##2t &>(*this), indent);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

const expr2tc *expr2t::get_sub_expr(size_t idx) const
{
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_get_sub_expr(                                       \
      static_cast<const kind##2t &>(*this), idx);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

size_t expr2t::get_num_sub_exprs() const
{
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_get_num_sub_exprs(                                  \
      static_cast<const kind##2t &>(*this));
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

void expr2t::foreach_operand_impl_const(const_op_delegate &f) const
{
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_foreach_operand_impl_const(                         \
      static_cast<const kind##2t &>(*this), f);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
}

void expr2t::foreach_operand_impl(op_delegate &f)
{
  switch (expr_id)
  {
#define IREP2_EXPR(kind, _)                                                    \
  case kind##_id:                                                              \
    return esbmct::generic_foreach_operand_impl(                               \
      static_cast<kind##2t &>(*this), f);
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  case end_expr_id:
    break;
  }
  std::unreachable();
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

// Field-name tables consumed by generic_tostring. Indexed by the order in
// each expr kind's `fields` tuple.
std::string constant_int2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_fixedbv2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_floatbv2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_struct2t::field_names[esbmct::num_type_fields] =
  {"members", "", "", "", ""};
std::string constant_union2t::field_names[esbmct::num_type_fields] =
  {"init_field", "members", "", "", ""};
std::string constant_bool2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string constant_array2t::field_names[esbmct::num_type_fields] =
  {"members", "", "", "", ""};
std::string constant_array_of2t::field_names[esbmct::num_type_fields] =
  {"initializer", "", "", "", ""};
std::string constant_string2t::field_names[esbmct::num_type_fields] =
  {"value", "kind", "", "", ""};
std::string constant_vector2t::field_names[esbmct::num_type_fields] =
  {"members", "", "", "", ""};
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
std::string cmp_three_way2t::field_names[esbmct::num_type_fields] =
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
  {"value_1", "value_2", "value_3", "rounding_mode", "", ""};
std::string ieee_sqrt2t::field_names[esbmct::num_type_fields] =
  {"value", "rounding_mode", "", "", ""};
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
std::string pointer_capability2t::field_names[esbmct::num_type_fields] =
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
std::string member_ref2t::field_names[esbmct::num_type_fields] =
  {"member_name", "", "", "", ""};
std::string ptr_mem2t::field_names[esbmct::num_type_fields] =
  {"source_value", "member_pointer", "", "", ""};
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
std::string races_check2t::field_names[esbmct::num_type_fields] =
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
std::string code_decl2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_dead2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string code_printf2t::field_names[esbmct::num_type_fields] =
  {"operands", "kind", "", "", ""};
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
std::string code_ifthenelse2t::field_names[esbmct::num_type_fields] =
  {"cond", "then_case", "else_case", "", ""};
std::string code_while2t::field_names[esbmct::num_type_fields] =
  {"cond", "body", "", "", ""};
std::string code_for2t::field_names[esbmct::num_type_fields] =
  {"init", "cond", "iter", "body", ""};
std::string code_switch2t::field_names[esbmct::num_type_fields] =
  {"value", "body", "", "", ""};
std::string code_break2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string code_continue2t::field_names[esbmct::num_type_fields] =
  {"", "", "", "", ""};
std::string code_label2t::field_names[esbmct::num_type_fields] =
  {"label", "code", "", "", ""};
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
std::string capability_base2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string capability_top2t::field_names[esbmct::num_type_fields] =
  {"value", "", "", "", ""};
std::string forall2t::field_names[esbmct::num_type_fields] =
  {"symbol", "predicate", "", "", ""};
std::string exists2t::field_names[esbmct::num_type_fields] =
  {"symbol", "predicate", "", "", ""};
std::string isinstance2t::field_names[esbmct::num_type_fields] =
  {"value", "type", "", "", ""};
std::string hasattr2t::field_names[esbmct::num_type_fields] =
  {"value", "attr", "", "", ""};
std::string isnone2t::field_names[esbmct::num_type_fields] =
  {"lhs", "rhs", "", "", ""};
