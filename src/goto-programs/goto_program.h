#ifndef CPROVER_GOTO_PROGRAM_H
#define CPROVER_GOTO_PROGRAM_H

/*! \defgroup gr_goto_programs Goto programs */

#include <cassert>
#include <ostream>
#include <set>
#include <irep2/irep2_utils.h>
#include <util/location.h>
#include <util/namespace.h>
#include <util/std_code.h>

#define forall_goto_program_instructions(it, program)                          \
  for (goto_programt::instructionst::const_iterator it =                       \
         (program).instructions.begin();                                       \
       it != (program).instructions.end();                                     \
       it++)

#define Forall_goto_program_instructions(it, program)                          \
  for (goto_programt::instructionst::iterator it =                             \
         (program).instructions.begin();                                       \
       it != (program).instructions.end();                                     \
       it++)

typedef enum
{
  NO_INSTRUCTION_TYPE = 0,
  GOTO = 1,           // branch, possibly guarded
  ASSUME = 2,         // non-failing guarded self loop
  ASSERT = 3,         // assertions
  OTHER = 4,          // anything else
  SKIP = 5,           // just advance the PC
  LOCATION = 8,       // semantically like SKIP
  END_FUNCTION = 9,   // exit point of a function
  ATOMIC_BEGIN = 10,  // marks a block without interleavings
  ATOMIC_END = 11,    // end of a block without interleavings
  RETURN = 12,        // return from a function
  ASSIGN = 13,        // assignment lhs:=rhs
  DECL = 14,          // declare a local variable
  DEAD = 15,          // marks the end-of-live of a local variable
  FUNCTION_CALL = 16, // call a function
  THROW = 17,         // throw an exception
  CATCH = 18,         // catch an exception
  THROW_DECL = 19,    // list of throws that a function can throw
  THROW_DECL_END = 20 // end of throw declaration
} goto_program_instruction_typet;

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
  */
  inline goto_programt(const goto_programt &src)
  {
    // CBMC didn't permit copy-construction, instead requiring calling
    // copy_from instead. While explicit is better than implicit though,
    // the only implication of allowing this is the occasional performance
    // loss, which is best identified by a profiler.
    copy_from(src);
    update();
  }

  /*! \brief assignment operator
      \param[in] src an empty goto program
  */
  inline goto_programt &operator=(const goto_programt &src)
  {
    // DO NOT COPY ME! I HAVE POINTERS IN ME!
    instructions.clear();
    copy_from(src);
    update();
    return *this;
  }

  bool hide;

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

    /// Returns the first (and only) successor for the usual case of a single
    /// target
    targett get_target() const
    {
      assert(targets.size() == 1);
      return targets.front();
    }

    /// Sets the first (and only) successor for the usual case of a single
    /// target
    void set_target(targett t)
    {
      targets.clear();
      targets.push_back(t);
    }

    bool has_target() const
    {
      return !targets.empty();
    }

    //! goto target labels
    typedef std::list<irep_idt> labelst;
    labelst labels;

    // for k-induction
    bool inductive_step_instruction;

    bool inductive_assertion;

    // for slicer (assumptions only)
    bool sliceable;

    //! is this node a branch target?
    inline bool is_target() const
    {
      return target_number != unsigned(-1);
    }

    //! clear the node
    inline void clear(goto_program_instruction_typet _type)
    {
      type = _type;
      targets.clear();
      guard = gen_true_expr();
      code = expr2tc();
      inductive_step_instruction = false;
      inductive_assertion = false;
      sliceable = false;
    }

    inline void make_goto()
    {
      clear(GOTO);
    }

    inline void make_location(const locationt &l)
    {
      clear(LOCATION);
      location = l;
    }

    inline void make_return()
    {
      clear(RETURN);
    }

    inline void make_function_call(const expr2tc &_code)
    {
      clear(FUNCTION_CALL);
      code = _code;
    }

    inline void make_skip()
    {
      clear(SKIP);
    }

    inline void make_throw()
    {
      clear(THROW);
    }

    inline void make_catch()
    {
      clear(CATCH);
    }

    inline void make_throw_decl()
    {
      clear(THROW_DECL);
    }

    inline void make_throw_decl_end()
    {
      clear(THROW_DECL_END);
    }

    inline void make_assertion(const expr2tc &g)
    {
      clear(ASSERT);
      guard = g;
    }

    inline void make_assumption(const expr2tc &g)
    {
      clear(ASSUME);
      guard = g;
    }

    inline void make_assignment()
    {
      clear(ASSIGN);
    }

    inline void make_other()
    {
      clear(OTHER);
    }

    inline void make_decl()
    {
      clear(DECL);
    }

    inline void make_dead()
    {
      clear(DEAD);
    }

    inline void make_atomic_begin()
    {
      clear(ATOMIC_BEGIN);
    }

    inline void make_atomic_end()
    {
      clear(ATOMIC_END);
    }

    inline void make_goto(targett _target)
    {
      make_goto();
      targets.push_back(_target);
    }

    inline void make_goto(targett _target, const expr2tc &g)
    {
      make_goto(_target);
      guard = g;
    }

    inline bool is_goto() const
    {
      return type == GOTO;
    }

    inline bool is_return() const
    {
      return type == RETURN;
    }

    inline bool is_assign() const
    {
      return type == ASSIGN;
    }

    inline bool is_function_call() const
    {
      return type == FUNCTION_CALL;
    }

    inline bool is_throw() const
    {
      return type == THROW;
    }

    inline bool is_catch() const
    {
      return type == CATCH;
    }

    inline bool is_skip() const
    {
      return type == SKIP;
    }

    inline bool is_location() const
    {
      return type == LOCATION;
    }

    inline bool is_other() const
    {
      return type == OTHER;
    }

    inline bool is_decl() const
    {
      return type == DECL;
    }

    inline bool is_assume() const
    {
      return type == ASSUME;
    }

    inline bool is_assert() const
    {
      return type == ASSERT;
    }

    inline bool is_atomic_begin() const
    {
      return type == ATOMIC_BEGIN;
    }

    inline bool is_atomic_end() const
    {
      return type == ATOMIC_END;
    }

    inline bool is_end_function() const
    {
      return type == END_FUNCTION;
    }

    inline instructiont()
      : location(static_cast<const locationt &>(get_nil_irep())),
        type(NO_INSTRUCTION_TYPE),
        inductive_step_instruction(false),
        inductive_assertion(false),
        sliceable(false),
        location_number(0),
        loop_number(unsigned(0)),
        target_number(unsigned(-1))
    {
      guard = gen_true_expr();
    }

    inline instructiont(goto_program_instruction_typet _type)
      : location(static_cast<const locationt &>(get_nil_irep())),
        type(_type),
        inductive_step_instruction(false),
        inductive_assertion(false),
        sliceable(false),
        location_number(0),
        loop_number(unsigned(0)),
        target_number(unsigned(-1))
    {
      guard = gen_true_expr();
    }

    //! swap two instructions
    void swap(instructiont &instruction)
    {
      instruction.code.swap(code);
      instruction.location.swap(location);
      std::swap(instruction.type, type);
      instruction.guard.swap(guard);
      instruction.targets.swap(targets);
      instruction.function.swap(function);
      std::swap(
        inductive_step_instruction, instruction.inductive_step_instruction);
      std::swap(inductive_assertion, instruction.inductive_assertion);
      std::swap(instruction.loop_number, loop_number);
      std::swap(sliceable, instruction.sliceable);
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

    //! Id of the scope within which a variable is declared (i.e., DECL).
    //! It does not have a lot of meaning for other types of instructions.
    unsigned int scope_id = 0;

    //! Id of the parent scope for the current "scope_id".
    unsigned int parent_scope_id = 0;

    //! Returns true if the instruction is a backwards branch.
    bool is_backwards_goto() const
    {
      if (!is_goto())
        return false;

      for (auto target : targets)
        if (target->location_number <= location_number)
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
      bool show_location = true) const;
  };

  typedef std::list<class instructiont> instructionst;

  typedef instructionst::iterator targett;
  typedef instructionst::const_iterator const_targett;
  typedef std::list<targett> targetst;
  typedef std::list<const_targett> const_targetst;

  //! The list of instructions in the goto program
  instructionst instructions;

  void get_successors(const_targett target, const_targetst &successors) const;

  /// Insertion that preserves jumps to "target".
  void insert_swap(targett target)
  {
    assert(target != instructions.end());
    const auto next = std::next(target);
    instructions.insert(next, instructiont())->swap(*target);
  }

  //! Insertion that preserves jumps to "target".
  //! The instruction is destroyed.
  void insert_swap(targett target, instructiont &instruction)
  {
    insert_swap(target);
    target->swap(instruction);
  }

  //! Insertion that preserves jumps to "target".
  //! The program p is destroyed.
  void insert_swap(targett target, goto_programt &p)
  {
    assert(target != instructions.end());
    if (p.instructions.empty())
      return;
    insert_swap(target, p.instructions.front());
    auto next = std::next(target);
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
    instructions.splice(instructions.end(), p.instructions);
  }

  //! Inserts the given program at the given location.
  //! The program is destroyed.
  inline void destructive_insert(targett target, goto_programt &p)
  {
    instructions.splice(target, p.instructions);
  }

  //! Adds an instruction at the end.
  //! \return The newly added instruction.
  inline targett add_instruction()
  {
    instructions.emplace_back();
    targett tmp = instructions.end();
    return --tmp;
  }

  //! Adds an instruction of given type at the end.
  //! \return The newly added instruction.
  inline targett add_instruction(instructiont &instruction)
  {
    instructions.emplace_back(instruction);
    targett tmp = instructions.end();
    return --tmp;
  }

  //! Adds an instruction of given type at the end.
  //! \return The newly added instruction.
  inline targett add_instruction(goto_program_instruction_typet type)
  {
    instructions.emplace_back(type);
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

  /// Sets the `function` member of each instruction if not yet set
  /// Note that a goto program need not be a goto function and therefore,
  /// we cannot do this in update(), but only at the level of
  /// of goto_functionst where goto programs are guaranteed to be
  /// named functions.
  void update_instructions_function(const irep_idt &function_id)
  {
    for (auto &instruction : instructions)
    {
      if (instruction.function.empty())
      {
        instruction.function = function_id;
      }
    }
  }

  //! Compute the target numbers
  void compute_target_numbers();

  //! Compute location numbers
  void compute_location_numbers(unsigned &nr)
  {
    for (auto &instruction : instructions)
      instruction.location_number = nr++;
  }

  //! Compute location numbers
  inline void compute_location_numbers()
  {
    unsigned nr = 0;
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
  goto_programt() : hide(false)
  {
  }
  virtual ~goto_programt() = default;

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

  typedef std::set<irep_idt> decl_identifierst;
  /// get the variables in decl statements
  void get_decl_identifiers(decl_identifierst &decl_identifiers) const;
};

bool operator<(
  const goto_programt::const_targett i1,
  const goto_programt::const_targett i2);

struct const_target_hash
{
  std::size_t operator()(const goto_programt::const_targett t) const
  {
    using hash_typet = decltype(&(*t));
    return std::hash<hash_typet>{}(&(*t));
  }
};

/// Functor to check whether iterators from different collections point at the
/// same object.
struct pointee_address_equalt
{
  template <class A, class B>
  bool operator()(const A &a, const B &b) const
  {
    return &(*a) == &(*b);
  }
};

#endif
