/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_IREP_H
#define CPROVER_IREP_H

#include <vector>
#include <list>
#include <map>
#include <string>

#include <assert.h>

#define USE_DSTRING
#define SHARING

#include "dstring.h"

typedef dstring irep_idt;
typedef dstring irep_namet;
typedef dstring_hash irep_id_hash;

#define forall_irep(it, irep) \
  for(irept::subt::const_iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#define Forall_irep(it, irep) \
  for(irept::subt::iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#define forall_named_irep(it, irep) \
  for(irept::named_subt::const_iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#define Forall_named_irep(it, irep) \
  for(irept::named_subt::iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#include <iostream>

class typet;

class irept
{
public:
  typedef std::vector<irept> subt;
  //typedef std::list<irept> subt;
  
  typedef std::map<irep_namet, irept> named_subt;

  bool is_nil() const { return id()=="nil"; }
  bool is_not_nil() const { return id()!="nil"; }

  explicit irept(const irep_idt &_id);

  #ifdef SHARING
  inline irept():data(NULL)
  {
  }

  inline irept(const irept &irep):data(irep.data)
  {
    if(data!=NULL)
    {
      assert(data->ref_count!=0);
      data->ref_count++;
      #ifdef IREP_DEBUG
      std::cout << "COPY " << data << " " << data->ref_count << std::endl;
      #endif
    }
  }

  inline irept &operator=(const irept &irep)
  {
    dt *tmp;
    assert(&irep!=this); // check if we assign to ourselves

    #ifdef IREP_DEBUG
    std::cout << "ASSIGN\n";
    #endif

    tmp = data;
    data=irep.data;
    if(data!=NULL) data->ref_count++;
    remove_ref(tmp);
    return *this;
  }

  ~irept()
  {
    remove_ref(data);
    data=NULL;
  }
  #else
  irept()
  {
  }
  #endif

  inline const irep_idt &id() const
  { return read().data; }
  
  inline const std::string &id_string() const
  { return read().data.as_string(); }

  inline void id(const irep_idt &_data)
  { write().data=_data; }

  const irept &find(const irep_namet &name) const;
  irept &add(const irep_namet &name);

  const std::string &get_string(const irep_namet &name) const
  {
    return get(name).as_string();
  }
  
  const irep_idt &get(const irep_namet &name) const;
  bool get_bool(const irep_namet &name) const;

  inline void set(const irep_namet &name, const irep_idt &value)
  { add(name).id(value); }
  
  void set(const irep_namet &name, const long value);
  void set(const irep_namet &name, const irept &irep);
  void remove(const irep_namet &name);
  void move_to_sub(irept &irep);
  void move_to_named_sub(const irep_namet &name, irept &irep);

  typet &type() { return (typet &)(add("type")); }
  const typet &type() const { return (typet &)(find("type")); }

  const irep_idt identifier(void) const {
    return get("identifier");
  }

  const irep_idt width(void) const {
    return get("width");
  }

  void identifier(irep_idt ident) {
    set("identifier", ident);
  }

  friend bool operator==(const irept &i1, const irept &i2);
   
  friend inline bool operator!=(const irept &i1, const irept &i2)
  { return !(i1==i2); }

  friend std::ostream& operator<< (std::ostream& out, const irept &irep);
  
  std::string to_string() const;
  
  void swap(irept &irep)
  {
    std::swap(irep.data, data);
  }

  friend bool operator<(const irept &i1, const irept &i2);
  friend bool ordering(const irept &i1, const irept &i2);

  int compare(const irept &i) const;
  
  void clear();

  void make_nil() { clear(); id("nil"); }
  
  subt &get_sub() { return write().sub; } // DANGEROUS
  const subt &get_sub() const { return read().sub; }
  named_subt &get_named_sub() { return write().named_sub; } // DANGEROUS
  const named_subt &get_named_sub() const { return read().named_sub; }
  named_subt &get_comments() { return write().comments; } // DANGEROUS
  const named_subt &get_comments() const { return read().comments; }
  
  size_t hash() const;
  size_t full_hash() const;
  
  friend bool full_eq(const irept &a, const irept &b);
  
  std::string pretty(unsigned indent=0) const;
  
protected:
  static bool is_comment(const irep_namet &name)
  { return !name.empty() && name[0]=='#'; }

public:
  class dt
  {
  public:
    #ifdef SHARING
    unsigned ref_count;
    #endif

    dstring data;

    named_subt named_sub;
    named_subt comments;
    subt sub;

    void clear()
    {
      data.clear();
      sub.clear();
      named_sub.clear();
      comments.clear();
    }
    
    void swap(dt &d)
    {
      d.data.swap(data);
      d.sub.swap(sub);
      d.named_sub.swap(named_sub);
      d.comments.swap(comments);
    }
    
    #ifdef SHARING
    dt():ref_count(1)
    {
    }
    #else
    dt()
    {
    }
    #endif
  };
  
protected:
  #ifdef SHARING
  dt *data;
  
  void remove_ref(dt *old_data);  
  
  const dt &read() const;

  inline dt &write()
  {
    detatch();
    return *data;
  }
  
  void detatch();
  #else
  dt data;
  
  inline const dt &read() const
  {
    return data;
  }

  inline dt &write()
  {
    return data;
  }
  #endif
};

extern inline const std::string &id2string(const irep_idt &d)
{
  return d.as_string();
}

extern inline const std::string &name2string(const irep_namet &n)
{
  return n.as_string();
}

struct irep_hash
{
  size_t operator()(const irept &irep) const { return irep.hash(); }
};

struct irep_full_hash
{
  size_t operator()(const irept &irep) const { return irep.full_hash(); }
};

struct irep_full_eq
{
  bool operator()(const irept &i1, const irept &i2) const 
  {
    return full_eq(i1, i2);
  }
};

const irept &get_nil_irep();

#endif
