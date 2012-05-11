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
    virtual const irep_idt get_original_name(const irep_idt &identifier,
                                             std::string idxname) const;
    virtual const irep_idt get_original_name(const irep_idt &identifier)const=0;
    virtual void get_original_name(exprt &expr) const;
    virtual void rename(expr2tc &expr)=0;
    virtual void remove(const irep_idt &identifier)=0;

    virtual std::string get_ident_name(const irep_idt &identifier) const=0;

    virtual ~renaming_levelt() { }
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
    virtual std::string get_ident_name(const irep_idt &identifier) const;
    virtual void remove(const irep_idt &identifier) { current_names.erase(identifier); }

    void rename(const irep_idt &identifier, unsigned frame)
    {
      current_names[identifier]=frame;
    }

    virtual const irep_idt get_original_name(const irep_idt &identifier) const
    {
      return renaming_levelt::get_original_name(identifier, "@");
    }

    virtual void get_original_name(exprt &expr) const
    {
      renaming_levelt::get_original_name(expr);
    }

    level1t() {}
    virtual ~level1t() { }

    virtual void print(std::ostream &out) const;
  };

  // level 2 -- SSA

  struct level2t:public renaming_levelt
  {
  protected:
    virtual void coveredinbees(const irep_idt &identifier, unsigned count, unsigned node_id);
  public:
    virtual irep_idt make_assignment(irep_idt ll1_identifier,
                                     const exprt &constant_value,
                                     const exprt &assigned_value);

    virtual void rename(expr2tc &expr);
    virtual void rename(const irep_idt &identifier, unsigned count)=0;

    virtual std::string get_ident_name(const irep_idt &identifier) const;
    virtual std::string name( const irep_idt &identifier, unsigned count) const;

    virtual void remove(const irep_idt &identifier)
    {
        current_names.erase(identifier);
    }

    virtual const irep_idt get_original_name(const irep_idt &identifier) const
    {
      return renaming_levelt::get_original_name(identifier, std::string("&"));
    }

    virtual void get_original_name(exprt &expr) const
    {
      renaming_levelt::get_original_name(expr);
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
