/*******************************************************************\
 Module: Expressions set container

 Author: Rafael Sá Menezes

 Date: April 2020

 Description:
  This file will contain a definition of a naive approach to save
  expressions, using the crc algorithm implemented in all expr with
  the std::set, the idea is to store an assertion as a set of sub-expr
  that must be true. This is then searched thorough a database to find
  a match and check it's satisfability.
\*******************************************************************/

#ifndef ESBMC_EXPR_SET_CONTAINER_H
#define ESBMC_EXPR_SET_CONTAINER_H

#include <set>
#include <cache/ssa_container.h>
#include <algorithm>
#include <memory>

// Typedefs
typedef long crc_hash;
typedef std::set<crc_hash> crc_expr;

/**
 * Stores a set of expressions representing the expr of an assertive
 * or guard assignment
 */
class expr_set_container : public ssa_container_item<crc_expr>
{
public:
  explicit expr_set_container(const crc_expr &other)
    : ssa_container_item<crc_expr>(other)
  {
  }
  /**
   * Checks if the expression contains all elements of @other
   * @param other
   * @return
   */
  virtual bool is_subset_of(const crc_expr &other);
};

typedef std::set<std::shared_ptr<expr_set_container>> ssa_container_type;

class ssa_set_container : public ssa_container<ssa_container_type>
{
public:
  ssa_set_container() : ssa_container<ssa_container_type>(){};

  // Check if the expression or a subexpression of it are present
  // at the container
  bool check(const crc_expr &items);

  // Add expression into the container
  void add(const crc_expr &items);

  /**
   * Caching is intended to run on a normal esbmc flow, however it may
   * be the case that this is used as a library such as testing
   */
  static void clear_cache();
};

/**
 * Base class to store a crc_set container
 */
class crc_set_storage : public container_storage<ssa_container_type>
{
public:
  crc_set_storage() : container_storage<ssa_container_type>(){};
  ssa_container<ssa_container_type> load(std::istream &) override = 0;
  void store(ssa_container<ssa_container_type> &) override = 0;
};

/**
 * Naive approach using a standard textual file in unix format
 *
 * This should make  simple to debug and verify algorithms during
 * the development
 */
class text_file_crc_set_storage : public crc_set_storage
{
public:
  text_file_crc_set_storage() : crc_set_storage()
  {
  }
  explicit text_file_crc_set_storage(std::string &output)
    : crc_set_storage(), filename(output)
  {
  }
  ssa_container<ssa_container_type> load();
  ssa_container<ssa_container_type> load(std::istream &) override;
  void store(ssa_container<ssa_container_type> &) override;

protected:
  std::string filename = "database";
};

#endif //ESBMC_EXPR_HASH_CONTAINER_H
