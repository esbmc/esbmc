#ifndef CPROVER_GOTO_SYMEX_WAYPOINT_H
#define CPROVER_GOTO_SYMEX_WAYPOINT_H

#include <big-int/bigint.hh>
#include <irep2/irep2.h>
#include <string>

#define c_nonset -1

class waypoint
{
public:
  enum Type
  {
    assumption,
    target,
    function_enter,
    function_return,
    branching,
    unknown
  };

  enum Action
  {
    follow, // must be passed exactly once
    avoid,  // must never be passed
    cycle   // must be passed infinitely
  };

  // Parsed \result constraint for function_return waypoints.
  // expr is an IRep2 expression tree with symbol2tc("\\result") as placeholder.
  struct parsed_constraintt
  {
    expr2tc expr;
    bool valid = false;
  };

  Type type = unknown;
  Action action = follow;
  size_t segment_idx = 0;
  std::string file;
  std::string value;
  std::string format;
  BigInt line = c_nonset;
  BigInt column = c_nonset;
  std::string function;
  irep_idt line_id;
  irep_idt column_id;
  irep_idt function_id;
  parsed_constraintt parsed_cond;
};

#endif
