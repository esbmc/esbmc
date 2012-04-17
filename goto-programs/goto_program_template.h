/*******************************************************************\

Module: Goto Program Template

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAM_TEMPLATE_H
#define CPROVER_GOTO_PROGRAM_TEMPLATE_H

#include <assert.h>

#include <iostream>
#include <set>

#include <namespace.h>

typedef enum { NO_INSTRUCTION_TYPE, GOTO, ASSUME, ASSERT, OTHER, SKIP,
               LOCATION, END_FUNCTION,
               ATOMIC_BEGIN, ATOMIC_END, RETURN, ASSIGN,
               FUNCTION_CALL, THROW, CATCH }
  goto_program_instruction_typet;
  
std::ostream &operator<<(std::ostream &, goto_program_instruction_typet);

template <class codeT, class guardT>
class goto_program_templatet
{
public:
  // DO NOT COPY ME! I HAVE POINTERS IN ME!
  goto_program_templatet(const goto_program_templatet &src)
  {
    assert(src.instructions.empty());
  }

  // DO NOT COPY ME! I HAVE POINTERS IN ME!
  goto_program_templatet &operator=(const goto_program_templatet &src)
  {
    assert(src.instructions.empty());
    instructions.clear();
    update();
    return *this;
  }

  // local variables
  typedef std::set<irep_idt> local_variablest;
  
  class instructiont
  {
  public:
    codeT code;
    
    // function this belongs to
    irep_idt function;
    
    // keep track of the location in the source file
    locationt location;
    
    // what kind of instruction?
    goto_program_instruction_typet type;

    // for gotos, assume, assert
    guardT guard;
    
    // for sync
    irep_idt event;

    // for gotos
    typedef typename std::list<class instructiont>::iterator targett;
    typedef typename std::list<class instructiont>::const_iterator const_targett;
    typedef std::list<targett> targetst;
    typedef std::list<const_targett> const_targetst;

    targetst targets;
    
    typedef std::list<irep_idt> labelst;
    labelst labels;

    std::set<targett> incoming_edges;
    
    void make_goto()
    {
      type=GOTO;
      targets.clear();
      guard.make_true();
      event="";
      code.make_nil();
    }
     
    void make_return()
    {
      type=RETURN;
      targets.clear();
      guard.make_true();
      event="";
      code.make_nil();
    }
     
    void make_skip()
    {
      type=SKIP;
      targets.clear();
      guard.make_true();
      event="";
      code.make_nil();
    }
     
    void make_assertion(const guardT &g)
    {
      type=ASSERT;
      targets.clear();
      guard=g;
      event="";
      code.make_nil();
    }
     
    void make_assumption(const guardT &g)
    {
      type=ASSUME;
      targets.clear();
      guard=g;
      event="";
      code.make_nil();
    }
     
    void make_goto(typename std::list<class instructiont>::iterator _target)
    {
      make_goto();
      targets.push_back(_target);
    }
    
    void make_other()
    {
      type=OTHER;
      targets.clear();
      guard.make_true();
      event="";
      code.make_nil();
    }
    
    void make_catch()
    {
      type=CATCH;
      targets.clear();
      guard.make_true();
      event="";
      code.make_nil();
    }

    // valid local variables at this point
    // this is obsolete and will be removed in future versions
    local_variablest local_variables;
    
    void add_local_variable(const irep_idt &id)
    {
      local_variables.insert(id);
    }   
    
    void add_local_variables(
      const local_variablest &locals)
    {
      local_variables.insert(locals.begin(), locals.end());
    }   
    
    void add_local_variables(
      const std::list<irep_idt> &locals)
    {
      for(typename std::list<irep_idt>::const_iterator
          it=locals.begin();
          it!=locals.end();
          it++)
        local_variables.insert(*it);
    }   
    
    inline bool is_goto         () const { return type==GOTO;          }
    inline bool is_return       () const { return type==RETURN;        }
    inline bool is_assign       () const { return type==ASSIGN;        }
    inline bool is_function_call() const { return type==FUNCTION_CALL; }
    inline bool is_skip         () const { return type==SKIP;          }
    inline bool is_location     () const { return type==LOCATION;      }
    inline bool is_other        () const { return type==OTHER;         }
    inline bool is_assume       () const { return type==ASSUME;        }
    inline bool is_assert       () const { return type==ASSERT;        }
    inline bool is_atomic_begin () const { return type==ATOMIC_BEGIN;  }
    inline bool is_atomic_end   () const { return type==ATOMIC_END;    }
    inline bool is_end_function () const { return type==END_FUNCTION;  }

    instructiont():
      location(static_cast<const locationt &>(get_nil_irep())),
      type(NO_INSTRUCTION_TYPE),
      location_number(0)
    {
      guard.make_true();
    }

    instructiont(goto_program_instruction_typet _type):
      location(static_cast<const locationt &>(get_nil_irep())),
      type(_type),
      location_number(0)
    {
      guard.make_true();
    }
    
    void swap(instructiont &instruction)
    {
      instruction.code.swap(code);
      instruction.location.swap(location);
      std::swap(instruction.type, type);
      instruction.guard.swap(guard);
      instruction.event.swap(event);
      instruction.targets.swap(targets);
      instruction.local_variables.swap(local_variables);
      instruction.function.swap(function);
    }
    
    // a globally unique number to identify a program location
    // it's guaranteed to be ordered in program order within
    // one goto_program
    unsigned location_number;
    
    // a globally unique number to identify loops
    unsigned loop_number;

    bool is_backwards_goto() const
    {
      if(!is_goto()) return false;

      for(typename targetst::const_iterator
          it=targets.begin();
          it!=targets.end();
          it++)
        if((*it)->location_number<=location_number)
          return true;
          
      return false;
    }    

    bool operator<(const class instructiont i1) const
    {
      if (function < i1.function)
        return true;

      if (location_number < i1.location_number)
        return true;

      return false;
    }
  };

  typedef std::list<class instructiont> instructionst;
  
  typedef typename instructionst::iterator targett;
  typedef typename instructionst::const_iterator const_targett;
  typedef typename std::list<targett> targetst;
  typedef typename std::list<const_targett> const_targetst;

  instructionst instructions;
  
  bool has_local_variable(
    class instructiont &instruction,
    const irep_idt &identifier)
  {
    return instruction.local_variables.count(identifier)!=0;
  }

  void get_successors(
    targett target,
    targetst &successors)
  {
    successors.clear();
    if(target==instructions.end()) return;

    targett next=target;
    next++;
    
    const instructiont &i=*target;
  
    if(i.is_goto())
    {
      for(typename targetst::const_iterator
          t_it=i.targets.begin();
          t_it!=i.targets.end();
          t_it++)
        successors.push_back(*t_it);
  
      if(!i.guard.is_true())
        successors.push_back(next);
    }
    else if(i.is_return())
    {
      // the successor is the end_function at the end
      successors.push_back(--instructions.end());
    }
    else
      successors.push_back(next);
  }

  void get_successors(
    const_targett target,
    const_targetst &successors) const
  {
    successors.clear();
    if(target==instructions.end()) return;

    const_targett next=target;
    next++;
    
    const instructiont &i=*target;
  
    if(i.is_goto())
    {
      for(typename targetst::const_iterator
          t_it=i.targets.begin();
          t_it!=i.targets.end();
          t_it++)
        successors.push_back(*t_it);
  
      if(!i.guard.is_true())
        successors.push_back(next);
    }
    else if(i.is_return())
    {
      // the successor is the end_function at the end
      successors.push_back(--instructions.end());
    }
    else
      successors.push_back(next);
  }

  void compute_incoming_edges();

  // insertion that preserves jumps to "target"
  void insert_swap(targett target, instructiont &instruction)
  {
    targett next=target;
    next++;
    instructions.insert(next, instructiont())->swap(*target);
    target->swap(instruction);
  }
  
  // insertion that preserves jumps to "target"
  void insert_swap(targett target, goto_program_templatet<codeT, guardT> &p)
  {
    if(p.instructions.empty()) return;
    insert_swap(target, p.instructions.front());
    targett next=target;
    next++;
    p.instructions.erase(p.instructions.begin());
    instructions.splice(next, p.instructions);
  }
  
  targett insert(targett target)
  {
    return instructions.insert(target, instructiont());
  }
  
  inline void destructive_append(goto_program_templatet<codeT, guardT> &p)
  {
    instructions.splice(instructions.end(),
                        p.instructions);
  }

  inline void destructive_insert(
    targett target,
    goto_program_templatet<codeT, guardT> &p)
  {
    instructions.splice(target,
                        p.instructions);
  }

  targett add_instruction()
  {
    instructions.push_back(instructiont());
    return --instructions.end();
  }

  targett add_instruction(goto_program_instruction_typet type)
  {
    instructions.push_back(instructiont(type));
    return --instructions.end();
  }

  // these assume that the targets are computed and numbered
  std::ostream& output(
    const class namespacet &ns,
    const irep_idt &identifier,
    std::ostream &out) const;
  
  std::ostream& output(std::ostream &out) const
  {
    return output(namespacet(contextt()), "", out);
  }
  
  virtual std::ostream& output_instruction(
    const class namespacet &ns,
    const irep_idt &identifier,
    std::ostream &out,
    typename instructionst::const_iterator it,
    bool show_location = true, bool show_variables = false) const=0;

  // keep a list of the targets
  typedef typename std::map<const_targett, unsigned> target_numberst;
  target_numberst target_numbers;
  
  // compute the list
  void compute_targets();

  // number them
  void number_targets();
  
  // compute location numbers
  void compute_location_numbers(unsigned &nr)
  {
    for(typename instructionst::iterator
        it=instructions.begin();
        it!=instructions.end();
        it++)
      it->location_number=nr++;
  }
  
  // compute location numbers
  void compute_location_numbers()
  {
    unsigned nr=0;
    compute_location_numbers(nr);
  }
  
  // compute loop numbers
  void compute_loop_numbers(unsigned &nr)
  {
    for(typename instructionst::iterator
        it=instructions.begin();
        it!=instructions.end();
        it++)
      if(it->is_backwards_goto())
        it->loop_number=nr++;
  }
  
  // compute loop numbers
  void compute_loop_numbers()
  {
    unsigned nr=0;
    compute_loop_numbers(nr);
  }
  
  void update()
  {
    compute_targets();
    number_targets();
    compute_location_numbers();
  }
  
  // empty program?
  bool empty() const
  {
    return instructions.empty();
  }

  // constructor/destructor
  goto_program_templatet()
  {
  }

  virtual ~goto_program_templatet()
  {
  }
   
  void swap(goto_program_templatet<codeT, guardT> &program)
  {
    program.instructions.swap(instructions);
    program.target_numbers.swap(target_numbers);
  }
  
  void clear()
  {
    instructions.clear();
    target_numbers.clear();
  }
  
  void copy_from(const goto_program_templatet<codeT, guardT> &src);
  
  bool has_assertion() const;
}; 

#include <langapi/language_util.h>
#include <iomanip>

template <class codeT, class guardT>
std::ostream& goto_program_templatet<codeT, guardT>::output(
  const namespacet &ns,
  const irep_idt &identifier,
  std::ostream& out) const
{
  // output program

  for(typename instructionst::const_iterator
      it=instructions.begin();
      it!=instructions.end();
      it++)
    output_instruction(ns, identifier, out, it);

  return out;  
}

template <class codeT, class guardT>
void goto_program_templatet<codeT, guardT>::compute_targets()
{
  target_numbers.clear();
 
  // get the targets
  for(typename instructionst::const_iterator
      i_it=instructions.begin();
      i_it!=instructions.end();
      i_it++)
    if(i_it->is_goto())
      for(typename instructiont::targetst::const_iterator
          t_it=i_it->targets.begin();
          t_it!=i_it->targets.end();
          t_it++)
        target_numbers.insert(
          std::pair<targett, unsigned>
          (*t_it, 0));
}

// number them
template <class codeT, class guardT>
void goto_program_templatet<codeT, guardT>::number_targets()
{
  // number the targets
  unsigned cnt=0;

  for(typename instructionst::const_iterator
      it=instructions.begin();
      it!=instructions.end();
      it++)
  {
    typename target_numberst::iterator t_it=
      target_numbers.find(it);

    if(t_it!=target_numbers.end())
      t_it->second=++cnt;
  }
}

template <class codeT, class guardT>
void goto_program_templatet<codeT, guardT>::copy_from(
  const goto_program_templatet<codeT, guardT> &src)
{
  // Definitions for mapping between the two programs
  typedef std::map<const_targett, targett> targets_mappingt;
  targets_mappingt targets_mapping;

  clear();

  // Loop over program - 1st time collects targets and copy

  for(typename instructionst::const_iterator
      it=src.instructions.begin();
      it!=src.instructions.end();
      it++)
  {
    targett new_instruction=add_instruction();
    targets_mapping[it]=new_instruction;
    *new_instruction=*it;
  }

  // Loop over program - 2nd time updates targets
  
  for(typename instructionst::iterator
      it=instructions.begin();
      it!=instructions.end();
      it++)
  {
    for(typename instructiont::targetst::iterator
        t_it=it->targets.begin();
        t_it!=it->targets.end();
        t_it++)
    {
      typename targets_mappingt::iterator
        m_target_it=targets_mapping.find(*t_it);

      if(m_target_it==targets_mapping.end())
        throw "copy_from: target not found";

      *t_it=m_target_it->second;
    }
  }

  compute_targets();
  number_targets();
}

// number them
template <class codeT, class guardT>
bool goto_program_templatet<codeT, guardT>::has_assertion() const
{
  for(typename instructionst::const_iterator
      it=instructions.begin();
      it!=instructions.end();
      it++)
    if(it->is_assert() && !it->guard.is_true())
      return true;

  return false;
}

template <class codeT, class guardT>
void goto_program_templatet<codeT, guardT>::compute_incoming_edges()
{
  for(typename instructionst::iterator
      it=instructions.begin();
      it!=instructions.end();
      it++)
  {
    targetst successors;
  
    get_successors(it, successors);
    
    for(typename targetst::const_iterator
        s_it=successors.begin();
        s_it!=successors.end();
        s_it++)
    {
      targett t=*s_it;

      if(t!=instructions.end())
        t->incoming_edges.insert(it);
    }
  }
}

template <class codeT, class guardT>
bool operator<(const typename goto_program_templatet<codeT, guardT>::const_targett i1,
               const typename goto_program_templatet<codeT, guardT>::const_targett i2);

#endif
