#pragma once
#include <memory>
#include <type_traits>
#include <util/crypto_hash.h>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/std_types.h>

/* The functions below form a small, fixed catalogue of operations the
 * irep_methods2 fold expressions invoke on every field of every node:
 * pretty-printing, comparison, ordering, CRC, and SHA-1 ingestion.
 *
 * Most field types (bool, unsigned int, the small enums, BigInt,
 * fixedbvt, ieee_floatt, irep_idt) share an entirely mechanical
 * implementation: `operator==` for cmp, the (-1/0/1) trinary on
 * `operator<` for lt, `std::hash<T>` for crc, raw POD ingestion for
 * hash. A primary template covers these by default; load-bearing
 * cases (BigInt's sign-aware CRC, the std::vector<...> overloads, the
 * null-safe expr2tc/type2tc dispatch, irep_idt's interned-index
 * hashing, the dummy `*_ids` stubs that preserve the parent-class
 * id-mix invariant) are declared explicitly below and win overload
 * resolution.
 */

std::string type_to_string(const bool &thebool, int);

std::string type_to_string(const sideeffect_data::allockind &data, int);

std::string type_to_string(const unsigned int &theval, int);

std::string type_to_string(const constant_string_data::kindt &theval, int);

std::string type_to_string(const symbol_data::renaming_level &theval, int);

std::string type_to_string(const BigInt &theint, int);

std::string type_to_string(const fixedbvt &theval, int);

std::string type_to_string(const ieee_floatt &theval, int);

std::string type_to_string(const std::vector<expr2tc> &theval, int indent);

std::string type_to_string(const std::vector<type2tc> &theval, int indent);

std::string type_to_string(const std::vector<irep_idt> &theval, int indent);

std::string type_to_string(const expr2tc &theval, int indent);

std::string type_to_string(const type2tc &theval, int indent);

std::string type_to_string(const irep_idt &theval, int);

/* Primary comparison: structural equality. Picked up by trivially-
 * comparable field types (bool, unsigned int, the enums, BigInt,
 * fixedbvt, ieee_floatt, irep_idt, std::vector<irep_idt>, and the
 * `irep_container`-mediated expr2tc/type2tc/std::vector<expr2tc|
 * type2tc>). */
template <class T>
inline bool do_type_cmp(const T &side1, const T &side2)
{
  return side1 == side2;
}

/* Primary ordering: trinary on operator<. Same coverage as
 * do_type_cmp. Specialisations below for expr2tc/type2tc replace this
 * with a single ltchecked dispatch (vs the double `<` here) and for
 * BigInt with its native compare(). */
template <class T>
inline int do_type_lt(const T &side1, const T &side2)
{
  if (side1 < side2)
    return -1;
  if (side2 < side1)
    return 1;
  return 0;
}

/* Primary CRC: std::hash<T> for non-enums; for enums, cast to uint8_t
 * before hashing so the byte width matches the previous explicit
 * std::hash<uint8_t> overloads bit-for-bit (enum class without a
 * fixed underlying type defaults to int, so a naive
 * std::hash<underlying_type_t<T>> would change the hash value and
 * break wire compatibility with on-disk caches / state hashes).
 * Specialisations below for BigInt / fixedbvt / ieee_floatt /
 * expr2tc / type2tc / irep_idt / vectors / the *_ids dummies
 * override this. */
template <class T>
inline size_t do_type_crc(const T &theval)
{
  if constexpr (std::is_enum_v<T>)
    return std::hash<uint8_t>{}(static_cast<uint8_t>(theval));
  else
    return std::hash<T>{}(theval);
}

/* Primary SHA-1 ingestion: raw POD bytes. Specialisations below pick
 * up irep_idt (ingest interned id, not the underlying string), the
 * dispatched types (BigInt/fixedbvt/ieee_floatt route through BigInt
 * sign-aware ingestion; expr2tc/type2tc dispatch to the contained
 * node's hash()), and the *_ids dummies. */
template <class T>
inline void do_type_hash(const T &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

/* Explicit overloads: load-bearing specialisations for the field
 * types whose semantics differ from the primary templates above. */

bool do_type_cmp(const type2t::type_ids &, const type2t::type_ids &);
bool do_type_cmp(const expr2t::expr_ids &, const expr2t::expr_ids &);

int do_type_lt(const BigInt &side1, const BigInt &side2);
int do_type_lt(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2);
int do_type_lt(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2);
int do_type_lt(const expr2tc &side1, const expr2tc &side2);
int do_type_lt(const type2tc &side1, const type2tc &side2);
int do_type_lt(const type2t::type_ids &, const type2t::type_ids &);
int do_type_lt(const expr2t::expr_ids &, const expr2t::expr_ids &);

size_t do_type_crc(const BigInt &theint);
size_t do_type_crc(const fixedbvt &theval);
size_t do_type_crc(const ieee_floatt &theval);
size_t do_type_crc(const std::vector<expr2tc> &theval);
size_t do_type_crc(const std::vector<type2tc> &theval);
size_t do_type_crc(const std::vector<irep_idt> &theval);
size_t do_type_crc(const expr2tc &theval);
size_t do_type_crc(const type2tc &theval);
size_t do_type_crc(const irep_idt &theval);
size_t do_type_crc(const type2t::type_ids &i);
size_t do_type_crc(const expr2t::expr_ids &i);

void do_type_hash(const BigInt &theint, crypto_hash &hash);
void do_type_hash(const fixedbvt &theval, crypto_hash &hash);
void do_type_hash(const ieee_floatt &theval, crypto_hash &hash);
void do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash);
void do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash);
void do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash);
void do_type_hash(const expr2tc &theval, crypto_hash &hash);
void do_type_hash(const type2tc &theval, crypto_hash &hash);
void do_type_hash(const irep_idt &theval, crypto_hash &hash);
void do_type_hash(const type2t::type_ids &, crypto_hash &);
void do_type_hash(const expr2t::expr_ids &, crypto_hash &);
