#include <irep2/irep2_template_utils.h>
std::string type_to_string(const bool &thebool, int)
{
  return (thebool) ? "true" : "false";
}

std::string type_to_string(const sideeffect_data::allockind &data, int)
{
  return (data == sideeffect_data::allockind::malloc)          ? "malloc"
         : (data == sideeffect_data::allockind::realloc)       ? "realloc"
         : (data == sideeffect_data::allockind::alloca)        ? "alloca"
         : (data == sideeffect_data::allockind::cpp_new)       ? "cpp_new"
         : (data == sideeffect_data::allockind::cpp_new_arr)   ? "cpp_new_arr"
         : (data == sideeffect_data::allockind::nondet)        ? "nondet"
         : (data == sideeffect_data::allockind::va_arg)        ? "va_arg"
         : (data == sideeffect_data::allockind::function_call) ? "function_call"
                                                               : "unknown";
}

std::string type_to_string(const unsigned int &theval, int)
{
  char buffer[64];
  snprintf(buffer, 63, "%d", theval);
  return std::string(buffer);
}

std::string type_to_string(const constant_string_data::kindt &theval, int)
{
  switch (theval)
  {
  case constant_string_data::DEFAULT:
    return "default";
  case constant_string_data::WIDE:
    return "wide";
  case constant_string_data::UNICODE:
    return "unicode";
  }
  assert(0 && "Unrecognized constant_string_data::kindt enum value");
  abort();
}

std::string type_to_string(const symbol_data::renaming_level &theval, int)
{
  switch (theval)
  {
  case symbol_data::level0:
    return "Level 0";
  case symbol_data::level1:
    return "Level 1";
  case symbol_data::level2:
    return "Level 2";
  case symbol_data::level1_global:
    return "Level 1 (global)";
  case symbol_data::level2_global:
    return "Level 2 (global)";
  default:
    assert(0 && "Unrecognized renaming level enum");
    abort();
  }
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

bool do_type_cmp(const bool &side1, const bool &side2)
{
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(const unsigned int &side1, const unsigned int &side2)
{
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(
  const sideeffect_data::allockind &side1,
  const sideeffect_data::allockind &side2)
{
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(
  const constant_string_data::kindt &side1,
  const constant_string_data::kindt &side2)
{
  return side1 == side2;
}

bool do_type_cmp(
  const symbol_data::renaming_level &side1,
  const symbol_data::renaming_level &side2)
{
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(const BigInt &side1, const BigInt &side2)
{
  // BigInt has its own equality operator.
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(const fixedbvt &side1, const fixedbvt &side2)
{
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(const ieee_floatt &side1, const ieee_floatt &side2)
{
  return (side1 == side2) ? true : false;
}

bool do_type_cmp(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2)
{
  return (side1 == side2);
}

bool do_type_cmp(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2)
{
  return (side1 == side2);
}

bool do_type_cmp(
  const std::vector<irep_idt> &side1,
  const std::vector<irep_idt> &side2)
{
  return (side1 == side2);
}

bool do_type_cmp(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // Catch null
  else if (side1.get() == nullptr || side2.get() == nullptr)
    return false;
  else
    return (side1 == side2);
}

bool do_type_cmp(const type2tc &side1, const type2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // both null ptr check
  if (side1.get() == nullptr || side2.get() == nullptr)
    return false; // One of them is null, the other isn't
  return (side1 == side2);
}

bool do_type_cmp(const irep_idt &side1, const irep_idt &side2)
{
  return (side1 == side2);
}

bool do_type_cmp(const type2t::type_ids &, const type2t::type_ids &)
{
  return true; // Dummy field comparison.
}

bool do_type_cmp(const expr2t::expr_ids &, const expr2t::expr_ids &)
{
  return true; // Dummy field comparison.
}

int do_type_lt(const bool &side1, const bool &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

int do_type_lt(const unsigned int &side1, const unsigned int &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

int do_type_lt(
  const sideeffect_data::allockind &side1,
  const sideeffect_data::allockind &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

int do_type_lt(
  const constant_string_data::kindt &side1,
  const constant_string_data::kindt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

int do_type_lt(
  const symbol_data::renaming_level &side1,
  const symbol_data::renaming_level &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

int do_type_lt(const BigInt &side1, const BigInt &side2)
{
  // BigInt also has its own less than comparator.
  return side1.compare(side2);
}

int do_type_lt(const fixedbvt &side1, const fixedbvt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side1 > side2)
    return 1;
  return 0;
}

int do_type_lt(const ieee_floatt &side1, const ieee_floatt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side1 > side2)
    return 1;
  return 0;
}

int do_type_lt(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2)
{
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

int do_type_lt(
  const std::vector<irep_idt> &side1,
  const std::vector<irep_idt> &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
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
  if (*side1.get() == *side2.get())
    return 0; // Both may be null;
  else if (side1.get() == nullptr)
    return -1;
  else if (side2.get() == nullptr)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

int do_type_lt(const irep_idt &side1, const irep_idt &side2)
{
  if (side1 < side2)
    return -1;
  if (side2 < side1)
    return 1;
  return 0;
}

int do_type_lt(const type2t::type_ids &, const type2t::type_ids &)
{
  return 0; // Dummy field comparison
}

int do_type_lt(const expr2t::expr_ids &, const expr2t::expr_ids &)
{
  return 0; // Dummy field comparison
}

size_t do_type_crc(const bool &theval)
{
  return boost::hash<bool>()(theval);
}

void do_type_hash(const bool &thebool, crypto_hash &hash)
{
  if (thebool)
  {
    uint8_t tval = 1;
    hash.ingest(&tval, sizeof(tval));
  }
  else
  {
    uint8_t tval = 0;
    hash.ingest(&tval, sizeof(tval));
  }
}

size_t do_type_crc(const unsigned int &theval)
{
  return boost::hash<unsigned int>()(theval);
}

void do_type_hash(const unsigned int &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

size_t do_type_crc(const sideeffect_data::allockind &theval)
{
  return boost::hash<uint8_t>()(theval);
}

void do_type_hash(const sideeffect_data::allockind &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

size_t do_type_crc(const constant_string_data::kindt &theval)
{
  return boost::hash<uint8_t>()(theval);
}

void do_type_hash(const constant_string_data::kindt &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

size_t do_type_crc(const symbol_data::renaming_level &theval)
{
  return boost::hash<uint8_t>()(theval);
}

void do_type_hash(const symbol_data::renaming_level &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

size_t do_type_crc(const BigInt &theint)
{
  if (theint.is_zero())
    return boost::hash<uint8_t>()(0);

  size_t crc = 0;
  std::array<unsigned char, 256> buffer;
  if (theint.dump(buffer.data(), buffer.size()))
  {
    for (unsigned int i = 0; i < buffer.size(); i++)
      boost::hash_combine(crc, buffer[i]);
  }
  else
  {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return crc;
}

void do_type_hash(const BigInt &theint, crypto_hash &hash)
{
  // Zero has no data in bigints.
  if (theint.is_zero())
  {
    uint8_t val = 0;
    hash.ingest(&val, sizeof(val));
    return;
  }

  std::array<unsigned char, 256> buffer;
  if (theint.dump(buffer.data(), buffer.size()))
  {
    hash.ingest(buffer.data(), buffer.size());
  }
  else
  {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
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
    boost::hash_combine(crc, it->do_crc());

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
    boost::hash_combine(crc, it->do_crc());

  return crc;
}

void do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash)
{
  for (auto const &it : theval)
    it->hash(hash);
}

size_t do_type_crc(const std::vector<irep_idt> &theval)
{
  size_t crc = 0;
  for (auto const &it : theval)
    boost::hash_combine(crc, it.as_string());

  return crc;
}

void do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash)
{
  for (auto const &it : theval)
    hash.ingest((void *)it.as_string().c_str(), it.as_string().size());
}

size_t do_type_crc(const expr2tc &theval)
{
  if (theval.get() != nullptr)
    return theval->do_crc();
  return boost::hash<uint8_t>()(0);
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
  return boost::hash<uint8_t>()(0);
}

void do_type_hash(const type2tc &theval, crypto_hash &hash)
{
  if (theval.get() != nullptr)
    theval->hash(hash);
}

size_t do_type_crc(const irep_idt &theval)
{
  return boost::hash<std::string>()(theval.as_string());
}

void do_type_hash(const irep_idt &theval, crypto_hash &hash)
{
  hash.ingest((void *)theval.as_string().c_str(), theval.as_string().size());
}

size_t do_type_crc(const type2t::type_ids &i)
{
  return boost::hash<uint8_t>()(i);
}

void do_type_hash(const type2t::type_ids &, crypto_hash &)
{
  // Dummy field crc
}

size_t do_type_crc(const expr2t::expr_ids &i)
{
  return boost::hash<uint8_t>()(i);
}

void do_type_hash(const expr2t::expr_ids &, crypto_hash &)
{
  // Dummy field crc
}
