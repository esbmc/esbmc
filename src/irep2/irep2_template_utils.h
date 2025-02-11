#pragma once
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

bool do_type_cmp(const bool &side1, const bool &side2);

bool do_type_cmp(const unsigned int &side1, const unsigned int &side2);

bool do_type_cmp(
  const sideeffect_data::allockind &side1,
  const sideeffect_data::allockind &side2);

bool do_type_cmp(
  const constant_string_data::kindt &side1,
  const constant_string_data::kindt &side2);

bool do_type_cmp(
  const symbol_data::renaming_level &side1,
  const symbol_data::renaming_level &side2);

bool do_type_cmp(const BigInt &side1, const BigInt &side2);

bool do_type_cmp(const fixedbvt &side1, const fixedbvt &side2);

bool do_type_cmp(const ieee_floatt &side1, const ieee_floatt &side2);

bool do_type_cmp(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2);

bool do_type_cmp(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2);

bool do_type_cmp(
  const std::vector<irep_idt> &side1,
  const std::vector<irep_idt> &side2);

bool do_type_cmp(const expr2tc &side1, const expr2tc &side2);

bool do_type_cmp(const type2tc &side1, const type2tc &side2);

bool do_type_cmp(const irep_idt &side1, const irep_idt &side2);

bool do_type_cmp(const type2t::type_ids &, const type2t::type_ids &);

bool do_type_cmp(const expr2t::expr_ids &, const expr2t::expr_ids &);

int do_type_lt(const bool &side1, const bool &side2);

int do_type_lt(const unsigned int &side1, const unsigned int &side2);

int do_type_lt(
  const sideeffect_data::allockind &side1,
  const sideeffect_data::allockind &side2);

int do_type_lt(
  const constant_string_data::kindt &side1,
  const constant_string_data::kindt &side2);

int do_type_lt(
  const symbol_data::renaming_level &side1,
  const symbol_data::renaming_level &side2);

int do_type_lt(const BigInt &side1, const BigInt &side2);

int do_type_lt(const fixedbvt &side1, const fixedbvt &side2);

int do_type_lt(const ieee_floatt &side1, const ieee_floatt &side2);

int do_type_lt(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2);
int do_type_lt(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2);

int do_type_lt(
  const std::vector<irep_idt> &side1,
  const std::vector<irep_idt> &side2);

int do_type_lt(const expr2tc &side1, const expr2tc &side2);

int do_type_lt(const type2tc &side1, const type2tc &side2);

int do_type_lt(const irep_idt &side1, const irep_idt &side2);

int do_type_lt(const type2t::type_ids &, const type2t::type_ids &);

int do_type_lt(const expr2t::expr_ids &, const expr2t::expr_ids &);

size_t do_type_crc(const bool &theval);

void do_type_hash(const bool &thebool, crypto_hash &hash);

size_t do_type_crc(const unsigned int &theval);

void do_type_hash(const unsigned int &theval, crypto_hash &hash);

size_t do_type_crc(const sideeffect_data::allockind &theval);

void do_type_hash(const sideeffect_data::allockind &theval, crypto_hash &hash);

size_t do_type_crc(const constant_string_data::kindt &theval);

void do_type_hash(const constant_string_data::kindt &theval, crypto_hash &hash);

size_t do_type_crc(const symbol_data::renaming_level &theval);

void do_type_hash(const symbol_data::renaming_level &theval, crypto_hash &hash);

size_t do_type_crc(const BigInt &theint);

void do_type_hash(const BigInt &theint, crypto_hash &hash);

size_t do_type_crc(const fixedbvt &theval);

void do_type_hash(const fixedbvt &theval, crypto_hash &hash);

size_t do_type_crc(const ieee_floatt &theval);

void do_type_hash(const ieee_floatt &theval, crypto_hash &hash);

size_t do_type_crc(const std::vector<expr2tc> &theval);

void do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash);

size_t do_type_crc(const std::vector<type2tc> &theval);

void do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash);

size_t do_type_crc(const std::vector<irep_idt> &theval);

void do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash);

size_t do_type_crc(const expr2tc &theval);

void do_type_hash(const expr2tc &theval, crypto_hash &hash);

size_t do_type_crc(const type2tc &theval);

void do_type_hash(const type2tc &theval, crypto_hash &hash);

size_t do_type_crc(const irep_idt &theval);

void do_type_hash(const irep_idt &theval, crypto_hash &hash);

size_t do_type_crc(const type2t::type_ids &i);

void do_type_hash(const type2t::type_ids &, crypto_hash &);

size_t do_type_crc(const expr2t::expr_ids &i);

void do_type_hash(const expr2t::expr_ids &, crypto_hash &);
