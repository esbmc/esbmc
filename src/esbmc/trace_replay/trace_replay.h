#pragma once

#include <string>

class namespacet;
class goto_tracet;

void replay_goto_trace(
  const namespacet &ns,
  const goto_tracet &trace,
  const std::string &context,
  bool proof_model);
