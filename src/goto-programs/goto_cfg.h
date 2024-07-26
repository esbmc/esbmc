#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string_view>
#include <goto-programs/goto_program.h>
#include <goto-programs/goto_functions.h>

/**
 * @brief An implementation of a control flow graph for goto programs.
 *
 * This class manipulates and transform a goto program by using a CFG abstraction.
 */
class goto_cfg
{
public:
  goto_cfg(goto_functionst &goto_functions);

  /**
     * @brief Generates a dot file containing the CFG.
     *
     * @param filename output file name
     */
  void dump_graph() const;

  /**
     * @brief A basic block is a sequence of instructions that has no branches in it.
     *
     * It consists of a sequence of instructions until a leader is found.
     * A leader consists in operations that create a new basic block,
     * i.e., label, if-goto, return, throw, catch, etc.
     */
  struct basic_block
  {
    enum class terminator_type
    {
      OTHER,
      IF_GOTO
    };
    goto_programt::instructionst::iterator begin;
    goto_programt::instructionst::iterator end;
    std::set<std::shared_ptr<basic_block>> successors;
    std::unordered_set<std::shared_ptr<basic_block>> predecessors;
    terminator_type terminator = terminator_type::OTHER;
  };

  std::unordered_map<std::string, std::vector<std::shared_ptr<basic_block>>>
    basic_blocks;

  template <class F>
   static void foreach_bb(const std::shared_ptr<basic_block> &start, F);

  struct Dominator
  {
    using Node = std::shared_ptr<basic_block>;
    const Node &start;
    Dominator(const Node &start) : start(start) { compute_dominators(); } 

    // Get dominators of a node
    inline std::unordered_set<Node> dom(std::shared_ptr<basic_block> &node) const
    {
      return dominators.at(node);
    }

    // Evaluates whether n1 dom n2
    inline bool dom(Node &n1, Node &n2) const
    {
      return dom(n2).count(n1);
    }
    
    // Returns the immediate dominator of n
    Node idom(const Node &n) const;

    // Computes the dominator frontier for node
    std::unordered_set<Node>
    dom_frontier(const Node &node) const;

    std::pair<Node, std::unordered_map<Node, std::unordered_set<Node>>>
    dom_tree() const;

    void dump_dominators() const;

  private:
    void compute_dominators();
    std::unordered_map<
    std::shared_ptr<basic_block>,
    std::unordered_set<std::shared_ptr<basic_block>>> dominators;
  };  
};




