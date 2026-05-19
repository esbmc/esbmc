#include <irep2/irep2_template_utils.h>
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

// Trivial do_type_cmp overloads (bool, unsigned int, the small enums,
// BigInt, fixedbvt, ieee_floatt, std::vector<expr2tc|type2tc|irep_idt>,
// expr2tc, type2tc, irep_idt) are covered by the primary template
// `template <class T> bool do_type_cmp(const T &, const T &)` in
// irep2_template_utils.h, which forwards to `operator==`. The two
// overloads below are dummies: the recursive walk only reaches a
// type_ids / expr_ids field when the parent base class has already
// short-circuited equality on the id, so the answer is invariantly
// true and we save an enum compare per call.

bool do_type_cmp(const type2t::type_ids &, const type2t::type_ids &)
{
  return true;
}

bool do_type_cmp(const expr2t::expr_ids &, const expr2t::expr_ids &)
{
  return true;
}

// Trivial do_type_lt overloads (bool, unsigned int, the small enums,
// fixedbvt, ieee_floatt, irep_idt, std::vector<irep_idt>) are covered
// by the primary template `template <class T> int do_type_lt(const T &,
// const T &)` in irep2_template_utils.h, which returns the trinary
// (-1/0/1) of operator<. The non-trivial cases below need their own
// dispatch.

int do_type_lt(const BigInt &side1, const BigInt &side2)
{
  // BigInt has a native compare() that already returns the trinary,
  // saving one operator< call vs the primary template.
  return side1.compare(side2);
}

int do_type_lt(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2)
{
  if (side1.size() != side2.size())
    return side1.size() < side2.size() ? -1 : 1;

  std::vector<expr2tc>::const_iterator it2 = side2.begin();
  for (auto const &it : side1)
  {
    int tmp = it->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    ++it2;
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

  std::vector<type2tc>::const_iterator it2 = side2.begin();
  for (auto const &it : side1)
  {
    int tmp = it->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    ++it2;
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

int do_type_lt(const type2t::type_ids &, const type2t::type_ids &)
{
  return 0; // Dummy field comparison
}

int do_type_lt(const expr2t::expr_ids &, const expr2t::expr_ids &)
{
  return 0; // Dummy field comparison
}

// Trivial do_type_crc / do_type_hash overloads for bool, unsigned int
// and the small enums (sideeffect_data::allockind,
// constant_string_data::kindt, symbol_data::renaming_level) are
// covered by the primary templates in irep2_template_utils.h:
// std::hash<T> (or std::hash<underlying_type_t<T>> for enums) for crc
// and raw POD ingestion for hash. The two *_ids dummies and the rest
// of the catalogue (BigInt, fixedbvt, ieee_floatt, expr2tc, type2tc,
// irep_idt, vectors) keep their explicit bodies below.

// BigInt::dump writes only the magnitude (most-significant-byte first, left-
// padded with zeros) and reports false when the supplied buffer is too small.
// Try a stack buffer for the common case; on overflow, double a heap buffer
// until the dump succeeds. The sign byte is fed first so +x and -x do not
// collide.
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
    esbmct::hash_combine(crc, it->do_crc());

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
    esbmct::hash_combine(crc, it->do_crc());

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
    return theval->do_crc();
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
    return theval->do_crc();
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
