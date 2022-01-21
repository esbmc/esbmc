/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SPECC_PARSER_H
#define CPROVER_SPECC_PARSER_H

#include <ansi-c/ansi_c_parse_tree.h>
#include <cassert>
#include <util/expr.h>
#include <util/i2string.h>
#include <util/parser.h>
#include <util/message/default_message.h>
#include <util/message/fmt_message_handler.h>

typedef enum
{
  ANSI_C_UNKNOWN,
  ANSI_C_SYMBOL,
  ANSI_C_TYPEDEF,
  ANSI_C_TAG
} ansi_c_id_classt;

int yyansi_cparse();

class ansi_c_parsert : public parsert
{
public:
  ansi_c_parse_treet parse_tree;

  ansi_c_parsert() = default;

  bool parse() override
  {
    return yyansi_cparse();
  }

  void clear() override
  {
    parsert::clear();
    parse_tree.clear();

    // scanner
    string_literal.clear();
    tag_following = false;
    asm_block_following = false;
    parenthesis_counter = 0;

    // setup global scope
    scopes.clear();

    // this is the global scope
    scopes.emplace_back();
  }

  // internal state scanner
  std::string string_literal;
  bool tag_following;
  bool asm_block_following;
  unsigned parenthesis_counter;

  enum
  {
    LANGUAGE,
    EXPRESSION
  } grammar;
  enum
  {
    ANSI,
    GCC,
    MSC
  } mode;

  class identifiert
  {
  public:
    ansi_c_id_classt id_class;
    std::string base_name;
  };

  class scopet
  {
  public:
    typedef std::unordered_map<irep_idt, identifiert, irep_id_hash> name_mapt;
    name_mapt name_map;

    std::string prefix;

    unsigned compound_counter, anon_counter;

    scopet() : compound_counter(0), anon_counter(0)
    {
    }

    void swap(scopet &scope)
    {
      name_map.swap(scope.name_map);
      prefix.swap(scope.prefix);
      std::swap(compound_counter, scope.compound_counter);
    }

    void print(std::ostream &out) const;
  };

  typedef std::list<scopet> scopest;
  scopest scopes;

  scopet &root_scope()
  {
    return scopes.front();
  }

  const scopet &root_scope() const
  {
    return scopes.front();
  }

  void pop_scope()
  {
    scopes.pop_back();
  }

  scopet &current_scope()
  {
    assert(!scopes.empty());
    return scopes.back();
  }

  static void
  convert_declarator(irept &declarator, const typet &type, irept &identifier);

  void new_declaration(
    const irept &type,
    irept &declarator,
    exprt &declaration,
    bool is_tag = false,
    bool put_into_scope = true);

  void move_declaration(exprt &expr)
  {
    parse_tree.declarations.emplace_back();
    parse_tree.declarations.back().swap(expr);
  }

  void copy_declaration(const exprt &expr)
  {
    parse_tree.declarations.push_back(to_ansi_c_declaration(expr));
  }

  void new_scope(const std::string &prefix)
  {
    const scopet &current = current_scope();
    scopes.emplace_back();
    scopes.back().prefix = current.prefix + prefix;
  }

  ansi_c_id_classt lookup(std::string &name, bool tag) const;

  static ansi_c_id_classt get_class(const typet &type);

  std::string get_anon_name()
  {
    std::string n = "#anon";
    n += i2string(current_scope().anon_counter++);
    return n;
  }
};

extern ansi_c_parsert ansi_c_parser;

int yyansi_cerror(const std::string &error);
void ansi_c_scanner_init();

#endif
