#pragma once

#include <optional>
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
    int uuid;
  };

  std::unordered_map<std::string, std::vector<std::shared_ptr<basic_block>>>
  basic_blocks;

  template <class F>
  static void foreach_bb(const std::shared_ptr<basic_block> &start, F);

  struct Dominator
  {

    using Node = std::shared_ptr<basic_block>;
    using DomTree =
      std::pair<Node, std::unordered_map<Node, std::unordered_set<Node>>>;

    const Node &start;

    
    Dominator(const Node &start) : start(start) { compute_dominators(); }

    // Evaluates whether n1 dom n2
    inline bool dom(const Node &n1, const Node &n2) const
    {
      return dom(n2).count(n1);
    }


    inline bool sdom(const Node &n1, const Node &n2) const
    {
      return n1 != n2 && dom(n1,n2);
    }

    // Returns the immediate dominator of n.  The idom of a
    // node n1 is the unique node n2 that n2 sdom n1 but does not sdom any other node that sdom n1. 
    Node idom(const Node &n) const;

    DomTree
    dom_tree() const;

    void dump_dominators() const;
    void dump_idoms() const;

    std::unordered_set<Node> dom_frontier(const Node &n) const;
    std::unordered_set<Node> dom_frontier(const std::unordered_set<Node> &nodes) const;
    std::unordered_set<Node>
    iterated_dom_frontier(const std::unordered_set<Node> &n) const;

    static std::unordered_map<Node, size_t> get_levels(const DomTree &dt);
    static std::unordered_set<Node> get_subtree(const DomTree &dt, const Node &n);

    struct DJGraph
    {
      const DomTree &tree;
      const goto_cfg::Dominator::Node &cfg;
      DJGraph(const DomTree &tree, const goto_cfg::Dominator::Node &cfg, const goto_cfg::Dominator &dom);
      using Graph =
        std::unordered_map<Node, std::unordered_set<Node>>;
      Graph _graph;
      Graph _jedges;
      Graph _dedges;

      
      

      void dump() const;
    };

    std::shared_ptr<DJGraph> dj;

  private:
    
    void compute_dominators();
    std::unordered_map<
      std::shared_ptr<basic_block>,
      std::unordered_set<std::shared_ptr<basic_block>>>
    dominators;
    // Get dominators of a node
    inline std::unordered_set<Node> dom(const Node &node) const
    {
      return dominators.at(node);
    }



    

  };  
};
