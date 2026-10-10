#pragma once

#include <goto-programs/goto_program.h>

#include <functional>
#include <vector>

/// A statement recovered from the shape of a GOTO program.
struct structured_stmtt
{
  enum kindt
  {
    INSTRUCTION,
    /// A loop's back edge or the jump over an else part: only its label.
    ANCHOR,
    WHILE,
    DO_WHILE,
    FOREVER,
    IF,
    BREAK,
    CONTINUE,
    GOTO
  };

  kindt kind;
  /// Where the label goes; null for DO_WHILE and FOREVER.
  const goto_programt::instructiont *instruction = nullptr;
  expr2tc condition;
  std::vector<structured_stmtt> body;
  std::vector<structured_stmtt> otherwise;
  /// Loops: the 1-based loop number, its head and its back edge.
  unsigned loop = 0;
  goto_programt::const_targett head, latch;
  std::vector<goto_programt::const_targett> targets;
};

/// Whether [first, last) may become one C block.
using enclosablet = std::function<
  bool(goto_programt::const_targett first, goto_programt::const_targett last)>;

/// What cannot be recovered, or what @p enclosable rejects, stays gotos.
std::vector<structured_stmtt> recover_structure(
  const goto_programt &program,
  const enclosablet &enclosable = {});
