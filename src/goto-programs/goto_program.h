/*******************************************************************\

Module: Goto Program Template

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAM_H
#define CPROVER_GOTO_PROGRAM_H

/*! \defgroup gr_goto_programs Goto programs */

#include <cassert>
#include <ostream>
#include <set>
#include <util/irep2.h>
#include <util/location.h>
#include <util/namespace.h>
#include <util/std_code.h>

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

  // local variables
  typedef std::list<irep_idt> local_variablest;

  local_variablest local_variables;

  void add_local_variable(const irep_idt &id)
  {
    local_variables.push_front(id);
  }

  void add_local_variables(const local_variablest &locals)
  {
    local_variables.insert(local_variables.begin(), locals.begin(), locals.end());
  }

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
      guard = gen_true_expr();
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
      guard = gen_true_expr();
    }

    inline instructiont(goto_program_instruction_typet _type):
      location(static_cast<const locationt &>(get_nil_irep())),
      type(_type),
      inductive_step_instruction(false),
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
      bool show_location=true) const;
  };

  typedef std::list<class instructiont> instructionst;

  typedef instructionst::iterator targett;
  typedef instructionst::const_iterator const_targett;
  typedef std::list<targett> targetst;
  typedef std::list<const_targett> const_targetst;

  //! The list of instructions in the goto program
  instructionst instructions;

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
    program.local_variables.swap(local_variables);
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

  // Template for extracting instructions /from/ a goto program, to a type
  // abstract something else.
  template <typename OutList, typename ListAppender, typename OutElem,
            typename SetAttrObj, typename SetAttrNil>
  void extract_instructions(OutList &list,
                       ListAppender listappend, SetAttrObj setattrobj,
                       SetAttrNil setattrnil) const;

  // Template for extracting instructions /from/ a type abstract something,
  // to a goto program.
  template <typename InList, typename InElem, typename FetchElem,
            typename ElemToInsn, typename GetAttr, typename IsAttrNil>
  void inject_instructions(InList list,
                              unsigned int len, FetchElem fetchelem,
                              ElemToInsn elemtoinsn, GetAttr getattr,
                              IsAttrNil isattrnil);
};

bool operator<(const goto_programt::const_targett i1,
               const goto_programt::const_targett i2);

template <typename OutList, typename ListAppender, typename OutElem,
         typename SetAttrObj, typename SetAttrNil>
void
goto_programt::extract_instructions(OutList &list, ListAppender listappend,
    SetAttrObj setattrobj, SetAttrNil setattrnil) const
{
  std::vector<OutElem> py_obj_vec;
  std::set<goto_programt::const_targett> targets;
  std::map<goto_programt::const_targett, unsigned int> target_map;

  // Convert instructions into python objects -- store in python list, as well
  // as in an stl vector, for easy access by index. Collect a set of all the
  // target iterators that are used in this function as well.
  for (const goto_programt::instructiont &insn : instructions) {
    OutElem o(insn);
    listappend(list, o);
    py_obj_vec.push_back(o);

    if (!insn.targets.empty()) {
      assert(insn.targets.size() == 1 && "Insn with multiple targets");
      targets.insert(*insn.targets.begin());
    }
  }

  // Map target iterators to index positions in the instruction list. Their
  // positions is the structure that we'll map over to python.
  unsigned int i = 0;
  for (auto it = instructions.begin();
       it != instructions.end();
       it++, i++) {
    if (targets.find(it) != targets.end())
      target_map.insert(std::make_pair(it, i));
  }

  // Iterate back over all the instructions again, this time filling out the
  // target attribute for each corresponding python object. If there's no
  // target, set it to None, otherwise set it to a reference to the
  // corresponding other python object.
  i = 0;
  for (const goto_programt::instructiont &insn : instructions) {
    if (insn.targets.empty()) {
      // If there's no target, set the target attribute to None
      setattrnil(py_obj_vec[i]);
    } else {
      assert(insn.targets.size() == 1 && "Insn with multiple targets");
      auto it = *insn.targets.begin();
      auto target_it = target_map.find(it);
      assert(target_it != target_map.end());

      // Set target attr to be reference to the correspondingly indexed python
      // object.
      setattrobj(py_obj_vec[i], py_obj_vec[target_it->second]);
    }
    i++;
  }

  return;
}

template <typename InList, typename InElem, typename FetchElem,
         typename ElemToInsn, typename GetAttr, typename IsAttrNil>
void
goto_programt::inject_instructions(InList list,
    unsigned int len, FetchElem fetchelem, ElemToInsn elemtoinsn,
    GetAttr getattr, IsAttrNil isattrnil)
{
  // Reverse the get_instructions function: generate a list of instructiont's
  // that preserve the 'target' attribute relation.

  std::vector<InElem> py_obj_vec;
  std::vector<goto_programt::targett> obj_it_vec;
  std::map<InElem, unsigned int> target_map;

  instructions.clear();

  // Extract list into vector we can easily index, pushing the extracted C++
  // object into the goto_programt's instruction list. Later store a vector of
  // iterators into that list: we need the instructiont storage and it's
  // iterators to stay stable, while mapping the 'target' relation back from
  // python into C++.
  for (unsigned int i = 0; i < len; i++) {
    InElem item = fetchelem(list, i);
    py_obj_vec.push_back(item);
    instructions.push_back(elemtoinsn(item));

    // XXX -- the performance of the following may be absolutely terrible,
    // it's not clear whether there's an operator< for std::map to infer
    // anywhere here. Based on assumption that a POD comparison is done against
    // the contained python ptr.
    target_map.insert(std::make_pair(item, i));
  }

  for (auto it = instructions.begin(); it != instructions.end(); it++)
    obj_it_vec.push_back(it);

  // Now iterate over each pair of python/c++ instructiont objs looking at the
  // 'target' attribute. Update the corresponding 'target' field of the C++
  // object accordingly
  for (unsigned int i = 0; i < py_obj_vec.size(); i++) {
    auto target = getattr(py_obj_vec[i]);
    auto it = obj_it_vec[i];

    if (isattrnil(target)) {
      it->targets.clear();
    } else {
      // Record a target -- map object to index, and from there to a list iter
      auto map_it = target_map.find(target);
      // Python user is entirely entitled to plug an arbitary object in here,
      // in which case we explode. Could raise an exception, but I prefer to
      // fail fast & fail hard. This isn't something the user should handle
      // anyway, and it's difficult for us to clean up afterwards.
      if (map_it == target_map.end())
        throw "Target of instruction is not in list";

      auto target_list_it = obj_it_vec[map_it->second];
      it->targets.clear();
      it->targets.push_back(target_list_it);
    }
  }

  return;
}

#endif
