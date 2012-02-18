#ifndef _GOTO_SYMEX_RENAMING_H_
#define _GOTO_SYMEX_RENAMING_H_

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

  typedef std::map<irep_idt, irep_idt> original_identifierst;

  struct renaming_level_datat
  {
    original_identifierst original_identifiers;
  };

  struct renaming_levelt
  {
  public:
    virtual const irep_idt &get_original_name(const irep_idt &identifier) const;
    virtual void get_original_name(exprt &expr) const;
    virtual void rename(exprt &expr, unsigned node_id)=0;
    virtual void rename(typet &type, unsigned node_id);
    virtual void remove(const irep_idt &identifier)=0;

    virtual std::string operator()(const irep_idt &identifier, unsigned exec_node_id) const=0;

    virtual ~renaming_levelt() { }

    original_identifierst original_identifiers;
  };

  // level 1 -- function frames
  // this is to preserve locality in case of recursion

  struct level1_datat:public renaming_level_datat
  {
    typedef std::map<irep_idt, unsigned> current_namest; // variables and its function frame number
    current_namest current_names;
    unsigned int _thread_id;
  };

  struct level1t:public renaming_levelt
  {
  public:
    std::string name(const irep_idt &identifier, unsigned frame,
                     unsigned execution_node_id) const;

    typedef std::map<irep_idt, unsigned> current_namest; // variables and its function frame number
    current_namest current_names;
    unsigned int _thread_id;

    virtual void rename(exprt &expr, unsigned node_id);
    virtual void rename(typet &type, unsigned node_id) { renaming_levelt::rename(type,node_id); }
    virtual std::string operator()(const irep_idt &identifier, unsigned exec_node_id) const;
    virtual void remove(const irep_idt &identifier) { current_names.erase(identifier); }

    void rename(const irep_idt &identifier, unsigned frame, unsigned exec_node_id)
    {
      current_names[identifier]=frame;
      original_identifiers[name(identifier, frame, exec_node_id)]=identifier;
    }

    level1t() {}
    virtual ~level1t() { }

    virtual void print(std::ostream &out, unsigned node_id) const;
  };

 // level 2 -- SSA

  struct level2_datat:public renaming_level_datat
  {
    struct valuet
    {
      unsigned count;
      exprt constant;
      unsigned node_id;
      valuet():
        count(0),
        constant(static_cast<const exprt &>(get_nil_irep())),
        node_id(0)
      {
      }
    };

    typedef std::map<irep_idt, valuet> current_namest;
    current_namest current_names;
    typedef std::map<irep_idt, crypto_hash> current_state_hashest;
    current_state_hashest current_hashes;
  };

  struct level2t:public renaming_levelt
  {
  public:
    virtual void rename(exprt &expr, unsigned node_id);
    virtual void rename(typet &type, unsigned node_id) { renaming_levelt::rename(type,node_id); }
    virtual std::string operator()(const irep_idt &identifier, unsigned node_id) const;
    virtual std::string stupid_operator(const irep_idt &identifier, unsigned node_id) const;
    virtual void remove(const irep_idt &identifier)
    {
        current_names.erase(identifier);
    }

    struct valuet
    {
      unsigned count;
      exprt constant;
      unsigned node_id;
      valuet():
        count(0),
        constant(static_cast<const exprt &>(get_nil_irep())),
        node_id(0)
      {
      }
    };

    typedef std::map<irep_idt, valuet> current_namest;
    current_namest current_names;
    typedef std::map<irep_idt, crypto_hash> current_state_hashest;
    current_state_hashest current_hashes;

    crypto_hash generate_l2_state_hash() const;

    void rename(const irep_idt &identifier, unsigned count, unsigned node_id)
    {
      valuet &entry=current_names[identifier];
      entry.count=count;
      entry.node_id = node_id;
      original_identifiers[name(identifier, entry.count)]=identifier;
    }

    std::string name(
      const irep_idt &identifier, unsigned count) const
    {
        unsigned int n_id = 0;
      current_namest::const_iterator it =current_names.find(identifier);
      if(it != current_names.end())
          n_id = it->second.node_id;
      return id2string(identifier)+"&"+i2string(n_id)+"#"+i2string(count);

    }

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
    virtual ~level2t() { }

    virtual void print(std::ostream &out, unsigned node_id) const;
  };
}

#endif /* _GOTO_SYMEX_RENAMING_H_ */
