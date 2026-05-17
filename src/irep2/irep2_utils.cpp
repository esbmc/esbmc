#include <irep2/irep2_utils.h>
#include <irep2/irep2_dispatch.h>
#include <util/c_types.h>

void make_not(expr2tc &expr)
{
  if (is_true(expr))
  {
    expr = gen_false_expr();
    return;
  }

  if (is_false(expr))
  {
    expr = gen_true_expr();
    return;
  }

  expr2tc new_expr;
  if (is_not2t(expr))
    new_expr = to_not2t(expr).value;
  else
    new_expr = not2tc(expr);

  expr.swap(new_expr);
}

expr2tc conjunction(std::vector<expr2tc> cs)
{
  if (cs.empty())
    return gen_true_expr();

  expr2tc res = cs[0];
  for (std::size_t i = 1; i < cs.size(); ++i)
    res = and2tc(res, cs[i]);

  return res;
}

expr2tc disjunction(std::vector<expr2tc> cs)
{
  if (cs.empty())
    return gen_true_expr();

  expr2tc res = cs[0];
  for (std::size_t i = 1; i < cs.size(); ++i)
    res = or2tc(res, cs[i]);

  return res;
}

expr2tc gen_nondet(const type2tc &type)
{
  return sideeffect2tc(
    type,
    expr2tc(),
    expr2tc(),
    std::vector<expr2tc>(),
    type2tc(),
    sideeffect2t::allockind::nondet);
}

expr2tc gen_zero(const type2tc &type, bool array_as_array_of)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return gen_false_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return constant_int2tc(type, BigInt(0));

  case type2t::fixedbv_id:
    return constant_fixedbv2tc(fixedbvt(fixedbv_spect(to_fixedbv_type(type))));

  case type2t::floatbv_id:
    return constant_floatbv2tc(
      ieee_floatt(ieee_float_spect(to_floatbv_type(type))));

  case type2t::vector_id:
  {
    auto vec_type = to_vector_type(type);
    assert(is_constant_int2t(vec_type.array_size));
    auto s = to_constant_int2t(vec_type.array_size);

    std::vector<expr2tc> members;
    for (long int i = 0; i < s.as_long(); i++)
      members.push_back(
        gen_zero(to_vector_type(type).subtype, array_as_array_of));

    return constant_vector2tc(type, members);
  }
  case type2t::array_id:
  {
    if (array_as_array_of)
      return constant_array_of2tc(type, gen_zero(to_array_type(type).subtype));

    auto arr_type = to_array_type(type);

    assert(is_constant_int2t(arr_type.array_size));
    auto s = to_constant_int2t(arr_type.array_size);

    std::vector<expr2tc> members;
    for (long int i = 0; i < s.as_long(); i++)
      members.push_back(
        gen_zero(to_array_type(type).subtype, array_as_array_of));

    return constant_array2tc(type, members);
  }

  case type2t::pointer_id:
    return symbol2tc(type, "NULL");

  case type2t::struct_id:
  {
    auto struct_type = to_struct_type(type);

    std::vector<expr2tc> members;
    for (auto const &member_type : struct_type.members)
      members.push_back(gen_zero(member_type, array_as_array_of));

    return constant_struct2tc(type, members);
  }

  case type2t::union_id:
  {
    auto union_type = to_union_type(type);

    assert(!union_type.members.empty());
    std::vector<expr2tc> members = {
      gen_zero(union_type.members.front(), array_as_array_of)};

    return constant_union2tc(type, union_type.member_names.front(), members);
  }

  default:
    break;
  }

  log_error("Can't generate zero for type {}", get_type_id(type));
  abort();
}

expr2tc gen_one(const type2tc &type)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return gen_true_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return constant_int2tc(type, BigInt(1));

  case type2t::fixedbv_id:
  {
    fixedbvt f(fixedbv_spect(to_fixedbv_type(type)));
    f.from_integer(BigInt(1));
    return constant_fixedbv2tc(f);
  }

  case type2t::floatbv_id:
  {
    ieee_floatt f(ieee_float_spect(to_floatbv_type(type)));
    f.from_integer(BigInt(1));
    return constant_floatbv2tc(f);
  }

  default:
    break;
  }

  log_error("Can't generate one for type {}", get_type_id(type));
  abort();
}

expr2tc distribute_vector_operation(
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

  auto result = is_op1_vector ? gen_zero(op1->type) : gen_zero(op2->type);

  for (size_t i = 0; i < to_constant_int2t(vector_length).as_ulong(); i++)
  {
    BigInt position(i);
    type2tc vector_type =
      to_vector_type(is_vector_type(op1->type) ? op1->type : op2->type).subtype;
    expr2tc local_op1 = op1;
    if (is_vector_type(op1->type))
    {
      local_op1 = index2tc(
        to_vector_type(op1->type).subtype,
        op1,
        constant_int2tc(get_uint32_type(), position));
    }

    expr2tc local_op2 = op2;
    if (op2 && is_vector_type(op2->type))
    {
      local_op2 = index2tc(
        to_vector_type(op2->type).subtype,
        op2,
        constant_int2tc(get_uint32_type(), position));
    }

    expr2tc to_add;
    switch (id)
    {
    case expr2t::neg_id:
      to_add = neg2tc(vector_type, local_op1);
      break;
    case expr2t::bitnot_id:
      to_add = bitnot2tc(vector_type, local_op1);
      break;
    case expr2t::sub_id:
      to_add = sub2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::mul_id:
      to_add = mul2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::div_id:
      to_add = div2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::modulus_id:
      to_add = modulus2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::add_id:
      to_add = add2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::shl_id:
      to_add = shl2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::bitxor_id:
      to_add = bitxor2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::bitor_id:
      to_add = bitor2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::bitand_id:
      to_add = bitand2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::ieee_add_id:
      to_add = ieee_add2tc(vector_type, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_div_id:
      to_add = ieee_div2tc(vector_type, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_sub_id:
      to_add = ieee_sub2tc(vector_type, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_mul_id:
      to_add = ieee_mul2tc(vector_type, local_op1, local_op2, rm);
      break;
    default:
      assert(0 && "Unsupported operation for Vector");
      abort();
    }
    to_constant_vector2t(result).datatype_members[i] = to_add;
  }
  return result;
}

expr2tc make_cmp_value(const type2tc &t, int v)
{
  if (!is_struct_type(t))
    return expr2tc();
  const struct_type2t &st = to_struct_type(t);
  if (st.members.empty())
    return expr2tc();
  std::vector<expr2tc> ops;
  ops.reserve(st.members.size());
  ops.push_back(constant_int2tc(st.members[0], BigInt(v)));
  for (size_t i = 1; i < st.members.size(); ++i)
    ops.push_back(gen_zero(st.members[i]));
  return constant_struct2tc(t, std::move(ops));
}

void get_symbols(
  const expr2tc &expr,
  std::unordered_set<expr2tc, irep2_hash> &symbols)
{
  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    symbol2t s = to_symbol2t(expr);
    if (s.thename.as_string().find("__ESBMC_") != std::string::npos)
      return;
    symbols.insert(expr);
  }

  expr->foreach_operand(
    [&symbols](const expr2tc &e) -> void { get_symbols(e, symbols); });
}

// Field-overload catalogue for the generic switch dispatchers in
// irep2_dispatch.h: pretty-printing, cmp/lt, CRC, SHA-1 ingestion,
// sub-expression iteration, and delegate calling, specialised for each
// field type that needs non-default behaviour. Primary templates live in
// irep2_template_utils.h / irep2_templates.h.

std::string indent_str_irep2(unsigned int indent)
{
  return std::string(indent, ' ');
}

template <>
bool do_get_sub_expr<expr2tc>(
  const expr2tc &item,
  size_t idx,
  size_t &it,
  const expr2tc *&ptr)
{
  if (idx == it)
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
  size_t idx,
  size_t &it,
  const expr2tc *&ptr)
{
  if (idx < it + item.size())
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

template <>
size_t do_count_sub_exprs<const expr2tc>(const expr2tc &)
{
  return 1;
}

template <>
size_t
do_count_sub_exprs<const std::vector<expr2tc>>(const std::vector<expr2tc> &item)
{
  return item.size();
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
  for (const expr2tc &r : ref)
    f(r);
}

template <>
void call_expr_delegate<std::vector<expr2tc>, expr2t::op_delegate>(
  std::vector<expr2tc> &ref,
  expr2t::op_delegate &f)
{
  for (expr2tc &r : ref)
    f(r);
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
  for (const type2tc &r : ref)
    f(r);
}

template <>
void call_type_delegate<std::vector<type2tc>, type2t::subtype_delegate>(
  std::vector<type2tc> &ref,
  type2t::subtype_delegate &f)
{
  for (type2tc &r : ref)
    f(r);
}

// ---------------------------------------------------------------------------
// type_to_string overloads
// ---------------------------------------------------------------------------

std::string type_to_string(const bool &thebool, int)
{
  return (thebool) ? "true" : "false";
}

std::string type_to_string(const sideeffect_allockind &data, int)
{
  return (data == sideeffect_allockind::malloc)          ? "malloc"
         : (data == sideeffect_allockind::realloc)       ? "realloc"
         : (data == sideeffect_allockind::alloca)        ? "alloca"
         : (data == sideeffect_allockind::cpp_new)       ? "cpp_new"
         : (data == sideeffect_allockind::cpp_new_arr)   ? "cpp_new_arr"
         : (data == sideeffect_allockind::nondet)        ? "nondet"
         : (data == sideeffect_allockind::va_arg)        ? "va_arg"
         : (data == sideeffect_allockind::function_call) ? "function_call"
                                                         : "unknown";
}

std::string type_to_string(const unsigned int &theval, int)
{
  char buffer[64];
  snprintf(buffer, 63, "%d", theval);
  return std::string(buffer);
}

std::string type_to_string(const constant_string_kindt &theval, int)
{
  switch (theval)
  {
  case constant_string_kindt::DEFAULT:
    return "default";
  case constant_string_kindt::WIDE:
    return "wide";
  case constant_string_kindt::UNICODE:
    return "unicode";
  }
  assert(0 && "Unrecognized constant_string_kindt enum value");
  abort();
}

std::string type_to_string(const printf_kindt &theval, int)
{
  switch (theval)
  {
  case printf_kindt::PRINTF:
    return "printf";
  case printf_kindt::FPRINTF:
    return "fprintf";
  case printf_kindt::DPRINTF:
    return "dprintf";
  case printf_kindt::SPRINTF:
    return "sprintf";
  case printf_kindt::VFPRINTF:
    return "vfprintf";
  case printf_kindt::SNPRINTF:
    return "snprintf";
  }
  assert(0 && "Unrecognized printf_kindt enum value");
  abort();
}

std::string type_to_string(const symbol_renaming_level &theval, int)
{
  switch (theval)
  {
  case symbol_renaming_level::level0:
    return "Level 0";
  case symbol_renaming_level::level1:
    return "Level 1";
  case symbol_renaming_level::level2:
    return "Level 2";
  case symbol_renaming_level::level1_global:
    return "Level 1 (global)";
  case symbol_renaming_level::level2_global:
    return "Level 2 (global)";
  }
  assert(0 && "Unrecognized renaming level enum");
  abort();
}

std::string type_to_string(const BigInt &theint, int)
{
  char buffer[256], *buf;

  buf = theint.as_string(buffer, 256);
  return std::string(buf);
}

std::string type_to_string(const fixedbvt &theval, int)
{
  return theval.to_ansi_c_string();
}

std::string type_to_string(const ieee_floatt &theval, int)
{
  return theval.to_ansi_c_string();
}

std::string type_to_string(const std::vector<expr2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  for (auto const &it : theval)
  {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str_irep2(indent) + std::string(buffer) + ": " +
               it->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

std::string type_to_string(const std::vector<type2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  for (auto const &it : theval)
  {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str_irep2(indent) + std::string(buffer) + ": " +
               it->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

std::string type_to_string(const std::vector<irep_idt> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  for (auto const &it : theval)
  {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str_irep2(indent) + std::string(buffer) + ": " +
               it.as_string() + "\n";
    i++;
  }

  return astring;
}

std::string type_to_string(const expr2tc &theval, int indent)
{
  if (theval.get() != nullptr)
    return theval->pretty(indent + 2);
  return "";
}

std::string type_to_string(const type2tc &theval, int indent)
{
  if (theval.get() != nullptr)
    return theval->pretty(indent + 2);
  else
    return "";
}

std::string type_to_string(const irep_idt &theval, int)
{
  return theval.as_string();
}

// do_type_lt overloads. Trivial cases (bool, unsigned int, enums,
// fixedbvt, ieee_floatt, irep_idt, std::vector<irep_idt>) use the
// primary template in irep2_template_utils.h.

int do_type_lt(const BigInt &side1, const BigInt &side2)
{
  return side1.compare(side2);
}

int do_type_lt(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2)
{
  if (side1.size() != side2.size())
    return side1.size() < side2.size() ? -1 : 1;

  int tmp = 0;
  std::vector<expr2tc>::const_iterator it2 = side2.begin();
  for (auto const &it : side1)
  {
    tmp = it->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

int do_type_lt(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2)
{
  if (side1.size() < side2.size())
    return -1;
  else if (side1.size() > side2.size())
    return 1;

  int tmp = 0;
  std::vector<type2tc>::const_iterator it2 = side2.begin();
  for (auto const &it : side1)
  {
    tmp = it->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

int do_type_lt(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return 0; // Catch nulls
  else if (side1.get() == nullptr)
    return -1;
  else if (side2.get() == nullptr)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

int do_type_lt(const type2tc &side1, const type2tc &side2)
{
  if (side1.get() == side2.get())
    return 0; // Catch nulls
  else if (side1.get() == nullptr)
    return -1;
  else if (side2.get() == nullptr)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

// ---------------------------------------------------------------------------
// do_type_crc / do_type_hash overloads
// do_type_crc / do_type_hash overloads. Trivial cases (bool, unsigned
// int, small enums) use the primary templates in irep2_template_utils.h.

// BigInt::dump writes only the magnitude (most-significant-byte first,
// left-padded with zeros) and reports false on buffer-too-small. Try a
// stack buffer; on overflow, double a heap buffer until it succeeds.
// The sign byte is fed first so +x and -x do not collide.
namespace
{
template <typename Sink>
void feed_bigint(const BigInt &theint, Sink &&sink)
{
  // Always include the sign so +x and -x do not collide.
  const uint8_t sign = theint.is_positive() ? 1 : 0;
  sink(&sign, sizeof(sign));

  if (theint.is_zero())
    return;

  std::array<unsigned char, 256> stack_buf;
  if (theint.dump(stack_buf.data(), stack_buf.size()))
  {
    sink(stack_buf.data(), stack_buf.size());
    return;
  }

  std::vector<unsigned char> heap_buf(stack_buf.size() * 2);
  while (!theint.dump(heap_buf.data(), heap_buf.size()))
    heap_buf.resize(heap_buf.size() * 2);
  sink(heap_buf.data(), heap_buf.size());
}
} // namespace

size_t do_type_crc(const BigInt &theint)
{
  size_t crc = 0;
  feed_bigint(theint, [&](const unsigned char *data, size_t len) {
    for (size_t i = 0; i < len; ++i)
      esbmct::hash_combine(crc, data[i]);
  });
  return crc;
}

void do_type_hash(const BigInt &theint, crypto_hash &hash)
{
  feed_bigint(theint, [&](const unsigned char *data, size_t len) {
    hash.ingest(data, len);
  });
}

size_t do_type_crc(const fixedbvt &theval)
{
  return do_type_crc(BigInt(theval.to_ansi_c_string().c_str()));
}

void do_type_hash(const fixedbvt &theval, crypto_hash &hash)
{
  do_type_hash(BigInt(theval.to_ansi_c_string().c_str()), hash);
}

size_t do_type_crc(const ieee_floatt &theval)
{
  return do_type_crc(theval.pack());
}

void do_type_hash(const ieee_floatt &theval, crypto_hash &hash)
{
  do_type_hash(theval.pack(), hash);
}

size_t do_type_crc(const std::vector<expr2tc> &theval)
{
  size_t crc = 0;
  for (auto const &it : theval)
    esbmct::hash_combine(crc, it->crc());

  return crc;
}

void do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash)
{
  for (auto const &it : theval)
    it->hash(hash);
}

size_t do_type_crc(const std::vector<type2tc> &theval)
{
  size_t crc = 0;
  for (auto const &it : theval)
    esbmct::hash_combine(crc, it->crc());

  return crc;
}

void do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash)
{
  for (auto const &it : theval)
    it->hash(hash);
}

size_t do_type_crc(const std::vector<irep_idt> &theval)
{
  // irep_idt is an interned dstring: its hash() returns the stable
  // table index, unique per string identity within the process. Mix
  // that directly instead of looking up the std::string and hashing
  // its char array per element.
  size_t crc = 0;
  for (auto const &it : theval)
    esbmct::hash_combine(crc, it.hash());

  return crc;
}

void do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash)
{
  for (auto const &it : theval)
  {
    size_t id = it.hash();
    hash.ingest(&id, sizeof(id));
  }
}

size_t do_type_crc(const expr2tc &theval)
{
  if (theval.get() != nullptr)
    return theval->crc();
  return std::hash<uint8_t>{}(0);
}

void do_type_hash(const expr2tc &theval, crypto_hash &hash)
{
  if (theval.get() != nullptr)
    theval->hash(hash);
}

size_t do_type_crc(const type2tc &theval)
{
  if (theval.get() != nullptr)
    return theval->crc();
  return std::hash<uint8_t>{}(0);
}

void do_type_hash(const type2tc &theval, crypto_hash &hash)
{
  if (theval.get() != nullptr)
    theval->hash(hash);
}

size_t do_type_crc(const irep_idt &theval)
{
  return theval.hash();
}

void do_type_hash(const irep_idt &theval, crypto_hash &hash)
{
  size_t id = theval.hash();
  hash.ingest(&id, sizeof(id));
}

size_t do_type_crc(const type2t::type_ids &i)
{
  return std::hash<uint8_t>{}(i);
}

void do_type_hash(const type2t::type_ids &, crypto_hash &)
{
  // Dummy field crc
}

size_t do_type_crc(const expr2t::expr_ids &i)
{
  return std::hash<uint8_t>{}(i);
}

void do_type_hash(const expr2t::expr_ids &, crypto_hash &)
{
  // Dummy field crc
}
