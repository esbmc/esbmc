/*******************************************************************\

Module: Goto Program Template

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAM_H
#define CPROVER_GOTO_PROGRAM_H

#include <irep2.h>
#include <assert.h>
/*! \defgroup gr_goto_programs Goto programs */

#include <cassert>
#include <ostream>
#include <set>

#include <namespace.h>
#include <location.h>
#include <std_code.h>

#define forall_goto_program_instructions(it, program) \
  for(goto_programt::instructionst::const_iterator it=(program).instructions.begin(); \
      it!=(program).instructions.end(); it++)

#define Forall_goto_program_instructions(it, program) \
  for(goto_programt::instructionst::iterator it=(program).instructions.begin(); \
      it!=(program).instructions.end(); it++)

typedef enum { NO_INSTRUCTION_TYPE=0,
               GOTO=1,          // branch, possibly guarded
               ASSUME=2,        // non-failing guarded self loop
               ASSERT=3,        // assertions
               OTHER=4,         // anything else
               SKIP=5,          // just advance the PC
               LOCATION=8,      // semantically like SKIP
               END_FUNCTION=9,  // exit point of a function
               ATOMIC_BEGIN=10, // marks a block without interleavings
               ATOMIC_END=11,   // end of a block without interleavings
               RETURN=12,       // return from a function
               ASSIGN=13,       // assignment lhs:=rhs
               DECL=14,         // declare a local variable
               DEAD=15,         // marks the end-of-live of a local variable
               FUNCTION_CALL=16,// call a function
               THROW=17,        // throw an exception
               CATCH=18,        // catch an exception
               THROW_DECL=19,   // list of throws that a function can throw
               THROW_DECL_END=20// end of throw declaration
             }
  goto_program_instruction_typet;

std::ostream &operator<<(std::ostream &, goto_program_instruction_typet);

/*! \brief A generic container class for a control flow graph
           for one function, in the form of a goto-program
    \ingroup gr_goto_programs
*/
class goto_programt
{
public:
  /*! \brief copy constructor
      \param[in] src an empty goto program
      \remark Use copy_from to copy non-empty goto-programs
  */
  inline goto_programt(const goto_programt &src __attribute__((unused)))
  {
    // DO NOT COPY ME! I HAVE POINTERS IN ME!
    assert(src.instructions.empty());
  }

  /*! \brief assignment operator
      \param[in] src an empty goto program
      \remark Use copy_from to copy non-empty goto-programs
  */
  inline goto_programt &operator=(const goto_programt &src
                                  __attribute__((unused)))
  {
    // DO NOT COPY ME! I HAVE POINTERS IN ME!
    assert(src.instructions.empty());
    instructions.clear();
    update();
    return *this;
  }

  // local variables
  typedef std::set<irep_idt> local_variablest;

  /*! \brief Container for an instruction of the goto-program
  */
  class instructiont
  {
  public:
    expr2tc code;

    //! function this belongs to
    irep_idt function;

    //! the location of the instruction in the source file
    locationt location;

    //! what kind of instruction?
    goto_program_instruction_typet type;

    //! guard for gotos, assume, assert
    expr2tc guard;

    //! the target for gotos and for start_thread nodes
    typedef std::list<class instructiont>::iterator targett;
    typedef std::list<class instructiont>::const_iterator const_targett;
    typedef std::list<targett> targetst;
    typedef std::list<const_targett> const_targetst;

    targetst targets;

    //! goto target labels
    typedef std::list<irep_idt> labelst;
    labelst labels;

    // for k-induction
    bool inductive_step_instruction;

    //! is this node a branch target?
    inline bool is_target() const
    { return target_number!=unsigned(-1); }

    //! clear the node
    inline void clear(goto_program_instruction_typet _type)
    {
      type=_type;
      targets.clear();
      guard = true_expr;
      code = expr2tc();
      inductive_step_instruction = false;
    }

    inline void make_goto() { clear(GOTO); }
    inline void make_return() { clear(RETURN); }
    inline void make_function_call(const expr2tc &_code) { clear(FUNCTION_CALL); code=_code; }
    inline void make_skip() { clear(SKIP); }
    inline void make_throw() { clear(THROW); }
    inline void make_catch() { clear(CATCH); }
    inline void make_throw_decl() { clear(THROW_DECL); }
    inline void make_throw_decl_end() { clear(THROW_DECL_END); }
    inline void make_assertion(const expr2tc &g) { clear(ASSERT); guard=g; }
    inline void make_assumption(const expr2tc &g) { clear(ASSUME); guard=g; }
    inline void make_assignment() { clear(ASSIGN); }
    inline void make_other() { clear(OTHER); }
    inline void make_decl() { clear(DECL); }
    inline void make_dead() { clear(DEAD); }
    inline void make_atomic_begin() { clear(ATOMIC_BEGIN); }
    inline void make_atomic_end() { clear(ATOMIC_END); }

    inline void make_goto(std::list<class instructiont>::iterator _target)
    {
      make_goto();
      targets.push_back(_target);
    }

    inline void make_goto(std::list<class instructiont>::iterator _target,
                          const expr2tc &g)
    {
      make_goto(_target);
      guard=g;
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
      for(std::list<irep_idt>::const_iterator
          it=locals.begin();
          it!=locals.end();
          it++)
        local_variables.insert(*it);
    }

    inline bool is_goto         () const { return type==GOTO;          }
    inline bool is_return       () const { return type==RETURN;        }
    inline bool is_assign       () const { return type==ASSIGN;        }
    inline bool is_function_call() const { return type==FUNCTION_CALL; }
    inline bool is_throw        () const { return type==THROW; }
    inline bool is_catch        () const { return type==CATCH;         }
    inline bool is_skip         () const { return type==SKIP;          }
    inline bool is_location     () const { return type==LOCATION;      }
    inline bool is_other        () const { return type==OTHER;         }
    inline bool is_assume       () const { return type==ASSUME;        }
    inline bool is_assert       () const { return type==ASSERT;        }
    inline bool is_atomic_begin () const { return type==ATOMIC_BEGIN;  }
    inline bool is_atomic_end   () const { return type==ATOMIC_END;    }
    inline bool is_end_function () const { return type==END_FUNCTION;  }

    inline instructiont():
      location(static_cast<const locationt &>(get_nil_irep())),
      type(NO_INSTRUCTION_TYPE),
      inductive_step_instruction(false),
      location_number(0),
      loop_number(unsigned(0)),
      target_number(unsigned(-1))
    {
      guard = true_expr;
    }

    inline instructiont(goto_program_instruction_typet _type):
      location(static_cast<const locationt &>(get_nil_irep())),
      type(_type),
      inductive_step_instruction(false),
      location_number(0),
      loop_number(unsigned(0)),
      target_number(unsigned(-1))
    {
      guard = true_expr;
    }

    //! swap two instructions
    void swap(instructiont &instruction)
    {
      instruction.code.swap(code);
      instruction.location.swap(location);
      std::swap(instruction.type, type);
      instruction.guard.swap(guard);
      instruction.targets.swap(targets);
      instruction.local_variables.swap(local_variables);
      instruction.function.swap(function);
      std::swap(inductive_step_instruction, instruction.inductive_step_instruction);
    }

    //! A globally unique number to identify a program location.
    //! It's guaranteed to be ordered in program order within
    //! one goto_program.
    unsigned location_number;

    //! Number unique per function to identify loops
    unsigned loop_number;

    //! A number to identify branch targets.
    //! This is -1 if it's not a target.
    unsigned target_number;

    //! Returns true if the instruction is a backwards branch.
    bool is_backwards_goto() const
    {
      if(!is_goto()) return false;

      for(targetst::const_iterator
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

    void dump() const;

    void output_instruction(
      const class namespacet &ns,
      const irep_idt &identifier,
      std::ostream &out,
      bool show_location=true,
      bool show_variables=false) const;
  };

  typedef std::list<class instructiont> instructionst;

  typedef instructionst::iterator targett;
  typedef instructionst::const_iterator const_targett;
  typedef std::list<targett> targetst;
  typedef std::list<const_targett> const_targetst;

  //! The list of instructions in the goto program
  instructionst instructions;

  bool has_local_variable(
    class instructiont &instruction,
    const irep_idt &identifier)
  {
    return instruction.local_variables.count(identifier)!=0;
  }

  void get_successors(
    targett target,
    targetst &successors);

  void get_successors(
    const_targett target,
    const_targetst &successors) const;

  //! Insertion that preserves jumps to "target".
  //! The instruction is destroyed.
  void insert_swap(targett target, instructiont &instruction)
  {
    assert(target!=instructions.end());
    targett next=target;
    next++;
    instructions.insert(next, instructiont())->swap(*target);
    target->swap(instruction);
  }

  //! Insertion that preserves jumps to "target".
  //! The program p is destroyed.
  void insert_swap(targett target, goto_programt &p)
  {
    assert(target!=instructions.end());
    if(p.instructions.empty()) return;
    insert_swap(target, p.instructions.front());
    targett next=target;
    next++;
    p.instructions.erase(p.instructions.begin());
    instructions.splice(next, p.instructions);
  }

  //! Insertion before the given target
  //! \return newly inserted location
  inline targett insert(targett target)
  {
    return instructions.insert(target, instructiont());
  }

  //! Appends the given program, which is destroyed
  inline void destructive_append(goto_programt &p)
  {
    instructions.splice(instructions.end(),
                        p.instructions);
  }

  //! Inserts the given program at the given location.
  //! The program is destroyed.
  inline void destructive_insert(
    targett target,
    goto_programt &p)
  {
    instructions.splice(target,
                        p.instructions);
  }

  //! Adds an instruction at the end.
  //! \return The newly added instruction.
  inline targett add_instruction()
  {
    instructions.push_back(instructiont());
    targett tmp = instructions.end();
    return --tmp;
  }

  //! Adds an instruction of given type at the end.
  //! \return The newly added instruction.
  inline targett add_instruction(instructiont &instruction)
  {
    instructions.push_back(instructiont(instruction));
    targett tmp = instructions.end();
    return --tmp;
  }

  //! Adds an instruction of given type at the end.
  //! \return The newly added instruction.
  inline targett add_instruction(goto_program_instruction_typet type)
  {
    instructions.push_back(instructiont(type));
    targett tmp = instructions.end();
    return --tmp;
  }

  //! Output goto program to given stream
  void dump() const;

  //! Output goto-program to given stream
  std::ostream &output(
    const namespacet &ns,
    const irep_idt &identifier,
    std::ostream &out) const;

  //! Compute the target numbers
  void compute_target_numbers();

  //! Compute location numbers
  void compute_location_numbers(unsigned &nr)
  {
    for(instructionst::iterator
        it=instructions.begin();
        it!=instructions.end();
        it++)
      it->location_number=nr++;
  }

  //! Compute location numbers
  inline void compute_location_numbers()
  {
    unsigned nr=0;
    compute_location_numbers(nr);
  }

  //! Compute loop numbers
  void compute_loop_numbers(unsigned int &num);

  //! Update all indices
  void update();

  //! Is the program empty?
  inline bool empty() const
  {
    return instructions.empty();
  }

  //! Constructor
  goto_programt()
  {
  }

  virtual ~goto_programt()
  {
  }

  //! Swap the goto program
  inline void swap(goto_programt &program)
  {
    program.instructions.swap(instructions);
  }

  //! Clear the goto program
  inline void clear()
  {
    instructions.clear();
  }

  //! Copy a full goto program, preserving targets
  void copy_from(const goto_programt &src);

  //! Does the goto program have an assertion?
  bool has_assertion() const;
};

bool operator<(const goto_programt::const_targett i1,
               const goto_programt::const_targett i2);

#endif
