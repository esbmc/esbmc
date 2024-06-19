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
 * Methods in this class will be reflected in the goto program.
 *
 * For instance, removing an edge in the CFG will remove the corresponding goto instruction.
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

    // Algorithms

protected:
    void add_edge(size_t from, size_t to);
    void remove_edge(size_t from, size_t to);
    void add_node(size_t id);
    void remove_node(size_t id);

     /**
     * @brief A basic block is a sequence of instructions that has no branches in it.
     *
     * It is a sequence of instructions starting from the beginning of a function or a label
     * until a goto statement is found (or the end of the function).
     */
    struct basic_block
    {
        goto_programt::instructionst::iterator begin;
        goto_programt::instructionst::iterator end;
        std::unordered_set<std::shared_ptr<basic_block>> successors;
        std::unordered_set<std::shared_ptr<basic_block>> predecessors;
    };

    std::unordered_map<std::string, std::shared_ptr<basic_block>> functions;
};

