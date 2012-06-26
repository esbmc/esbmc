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
    virtual void rename(expr2tc &expr) const = 0;
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

    virtual void rename(expr2tc &expr) const;
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
    class name_record {
    public:
      name_record(const symbol2t &sym)
        : base_name(sym.thename), lev(sym.rlevel), l1_num(sym.level1_num),
          t_num(sym.thread_num)
      {
        hacky_hash h;
        h.ingest(base_name.get_no());
        h.ingest((uint8_t)lev);
        h.ingest(l1_num);
        h.ingest(t_num);
        hash = h.result();
      }

      int compare(const name_record &ref) const
      {
        if (hash < ref.hash)
          return -1;
        else if (hash > ref.hash)
          return 1;

        if (base_name < ref.base_name)
          return -1;
        else if (ref.base_name < base_name)
          return 1;

        if (lev < ref.lev)
          return -1;
        else if (lev > ref.lev)
          return 1;

        if (l1_num < ref.l1_num)
          return -1;
        else if (l1_num > ref.l1_num)
          return 1;

        if (t_num < ref.t_num)
          return -1;
        else if (t_num > ref.t_num)
          return 1;

        return 0;
      }

      bool operator<(const name_record &ref) const
      {
        if (compare(ref) == -1)
          return true;
        return false;
      }

      bool operator==(const name_record &ref) const
      {
        if (compare(ref) == 0)
          return true;
        return false;
      }

      irep_idt base_name;
      symbol2t::renaming_level lev;
      unsigned int l1_num;
      unsigned int t_num;

      // Not a part of comparisons etc,
      size_t hash;
    };


    virtual void coveredinbees(expr2tc &lhs_sym, unsigned count, unsigned node_id);
  public:
    virtual void make_assignment(expr2tc &lhs_symbol,
                                 const expr2tc &constant_value,
                                 const expr2tc &assigned_value);

    virtual void rename(expr2tc &expr) const;
    virtual void rename(expr2tc &expr, unsigned count)=0;

    virtual void get_ident_name(expr2tc &symbol) const;
    virtual std::string name( const irep_idt &identifier, unsigned count) const;

    virtual void remove(const expr2tc &symbol)
    {
        current_names.erase(symbol);
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

    void get_variables(std::set<expr2tc> &vars) const
    {
      for(current_namest::const_iterator it=current_names.begin();
          it!=current_names.end();
          it++)
      {
                  vars.insert(it->first);
      }
    }

    unsigned current_number(const expr2tc &sym) const;

    level2t() { };
    virtual ~level2t() { };
    virtual level2t *clone(void) const = 0;

    virtual void print(std::ostream &out) const;
    virtual void dump() const;

  protected:
    typedef std::map<const expr2tc, valuet> current_namest;
    current_namest current_names;
    typedef std::map<const expr2tc, crypto_hash> current_state_hashest;
    current_state_hashest current_hashes;
  };
}

#endif /* _GOTO_SYMEX_RENAMING_H_ */
