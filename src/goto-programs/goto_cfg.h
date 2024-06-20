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
     */
    struct basic_block
    {
        goto_programt::instructionst::iterator begin;
        goto_programt::instructionst::iterator end;
        std::unordered_set<std::shared_ptr<basic_block>> successors;
        std::unordered_set<std::shared_ptr<basic_block>> predecessors;
    };

    std::unordered_map<std::string, std::vector<std::shared_ptr<basic_block>>> basic_blocks;
};

