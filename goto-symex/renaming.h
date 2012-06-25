#ifndef _GOTO_SYMEX_RENAMING_H_
#define _GOTO_SYMEX_RENAMING_H_

#include <irep2.h>

#include <stdint.h>
#include <string.h>

#include <string>
#include <stack>
#include <vector>
#include <set>

#include <guard.h>
#include <expr_util.h>
#include <std_expr.h>
#include <i2string.h>

#include "crypto_hash.h"

namespace renaming {

  struct renaming_levelt
  {
  public:
    virtual void get_original_name(expr2tc &expr) const = 0;
    virtual void rename(expr2tc &expr)=0;
    virtual void remove(const expr2tc &symbol)=0;

    virtual void get_ident_name(expr2tc &symbol) const=0;

    virtual ~renaming_levelt() { }
  protected:
    void get_original_name(expr2tc &expr, symbol2t::renaming_level lev) const;
  };

  // level 1 -- function frames
  // this is to preserve locality in case of recursion

  struct level1t:public renaming_levelt
  {
  public:
    virtual std::string name(const irep_idt &identifier, unsigned frame) const;

    typedef std::map<irep_idt, unsigned> current_namest; // variables and its function frame number
    current_namest current_names;
    unsigned int _thread_id;

    virtual void rename(expr2tc &expr);
    virtual void get_ident_name(expr2tc &symbol) const;
    virtual void remove(const expr2tc &symbol)
    {
      current_names.erase(to_symbol2t(symbol).get_symbol_name());
    }

    void rename(const expr2tc &symbol, unsigned frame)
    {
      // Given that this is level1, use base symbol.
      current_names[to_symbol2t(symbol).thename]=frame;
    }

    virtual void get_original_name(expr2tc &expr) const
    {
      renaming_levelt::get_original_name(expr, symbol2t::level0);
    }

    level1t() {}
    virtual ~level1t() { }

    virtual void print(std::ostream &out) const;
  };

  // level 2 -- SSA

  struct level2t:public renaming_levelt
  {
  protected:
    virtual void coveredinbees(expr2tc &lhs_sym, unsigned count, unsigned node_id);
  public:
    virtual void make_assignment(expr2tc &lhs_symbol,
                                 const expr2tc &constant_value,
                                 const expr2tc &assigned_value);

    virtual void rename(expr2tc &expr);
    virtual void rename(expr2tc &expr, unsigned count)=0;

    virtual void get_ident_name(expr2tc &symbol) const;
    virtual std::string name( const irep_idt &identifier, unsigned count) const;

    virtual void remove(const expr2tc &symbol)
    {
        current_names.erase(to_symbol2t(symbol).get_symbol_name());
    }

    virtual void get_original_name(expr2tc &expr) const
    {
      renaming_levelt::get_original_name(expr, symbol2t::level1);
    }

    struct valuet
    {
      unsigned count;
      expr2tc constant;
      unsigned node_id;
      valuet():
        count(0),
        constant(),
        node_id(0)
      {
      }
    };

    typedef std::map<irep_idt, valuet> current_namest;
    current_namest current_names;
    typedef std::map<irep_idt, crypto_hash> current_state_hashest;
    current_state_hashest current_hashes;

    void get_variables(std::set<irep_idt> &vars) const
    {
      for(current_namest::const_iterator it=current_names.begin();
          it!=current_names.end();
          it++)
      {
                  vars.insert(it->first);
      }
    }

    unsigned current_number(const irep_idt &identifier) const;

    level2t() { };
    virtual ~level2t() { };
    virtual level2t *clone(void) const = 0;

    virtual void print(std::ostream &out) const;
    virtual void dump() const;
  };
}

#endif /* _GOTO_SYMEX_RENAMING_H_ */
