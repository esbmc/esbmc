#include <goto-programs/goto_structure.h>

#include <irep2/irep2_utils.h>

#include <map>
#include <optional>
#include <unordered_map>

namespace
{
using instructiont = goto_programt::instructiont;
using targett = goto_programt::const_targett;
using stmtst = std::vector<structured_stmtt>;

expr2tc negate(expr2tc cond)
{
  make_not(cond);
  return cond;
}

class structure_recoveryt
{
public:
  structure_recoveryt(
    const goto_programt &program,
    const enclosablet &enclosable)
    : instructions(program.instructions), enclosable(enclosable)
  {
    std::size_t n = 0;
    for (const instructiont &i : instructions)
      position.emplace(&i, n++);
    find_loops();
  }

  stmtst recover()
  {
    return range(instructions.begin(), instructions.end());
  }

private:
  struct loopt
  {
    targett head;
    targett latch;
    unsigned number;
  };

  struct open_loopt
  {
    targett next_iteration;
    targett after;
  };

  const goto_programt::instructionst &instructions;
  const enclosablet &enclosable;
  /// Deeper statements stay gotos, bounding the recursion.
  unsigned depth = 0;
  static constexpr unsigned max_depth = 256;
  std::unordered_map<const instructiont *, std::size_t> position;
  std::map<const instructiont *, loopt> loops;
  std::vector<open_loopt> open_loops;

  std::size_t at(targett it) const
  {
    return it == instructions.end() ? instructions.size() : position.at(&*it);
  }

  static structured_stmtt
  statement(structured_stmtt::kindt kind, targett it, expr2tc condition = {})
  {
    structured_stmtt s;
    s.kind = kind;
    s.instruction = &*it;
    s.condition = std::move(condition);
    return s;
  }

  bool can_enclose(targett first, targett last) const
  {
    return depth < max_depth && (!enclosable || enclosable(first, last));
  }

  struct nestingt
  {
    unsigned &depth;
    explicit nestingt(unsigned &depth) : depth(++depth)
    {
    }
    ~nestingt()
    {
      --depth;
    }
  };

  /// The `while` test, or the back edge if there is none.
  targett while_test(const loopt &loop) const
  {
    if (!is_true(loop.latch->guard))
      return loop.latch;
    targett cond = loop.head;
    while (cond != loop.latch && (cond->is_skip() || cond->is_location()))
      ++cond;
    const bool is_test = cond != loop.latch && cond->is_goto() &&
                         cond->targets.size() == 1 && !is_true(cond->guard) &&
                         cond->targets.front() == std::next(loop.latch);
    return is_test ? cond : loop.latch;
  }

  void find_loops()
  {
    for (targett it = instructions.begin(); it != instructions.end(); ++it)
    {
      if (!it->is_goto() || it->targets.size() != 1)
        continue;
      targett head = it->targets.front();
      if (at(head) >= at(it))
        continue;
      auto [loop, fresh] = loops.emplace(&*head, loopt{head, it, 0});
      if (!fresh && at(it) > at(loop->second.latch))
        loop->second.latch = it;
    }
    unsigned n = 0;
    for (const instructiont &i : instructions)
    {
      auto loop = loops.find(&i);
      if (loop != loops.end())
        loop->second.number = ++n;
    }
  }

  stmtst range(targett first, targett last)
  {
    stmtst out;
    for (targett it = first; it != last;)
      it = next_statement(it, last, out);
    return out;
  }

  targett next_statement(targett it, targett last, stmtst &out)
  {
    auto loop = loops.find(&*it);
    if (loop != loops.end() && at(loop->second.latch) < at(last))
    {
      const loopt &l = loop->second;
      const targett test = while_test(l);
      const targett first = test == l.latch ? l.head : std::next(test);
      if (can_enclose(first, std::next(l.latch)))
        return recover_loop(l, test, out);
    }
    if (it->is_goto())
      return recover_goto(it, last, out);
    out.push_back(statement(structured_stmtt::INSTRUCTION, it));
    return std::next(it);
  }

  targett recover_loop(const loopt &loop, targett test, stmtst &out)
  {
    const nestingt nesting(depth);
    const targett after = std::next(loop.latch);
    const bool do_while = !is_true(loop.latch->guard);

    structured_stmtt s;
    // A do-while `continue` runs the test; a jump to the head does not.
    open_loops.push_back({do_while ? loop.latch : loop.head, after});
    if (test != loop.latch)
    {
      for (targett it = loop.head; it != test; ++it)
        out.push_back(statement(structured_stmtt::INSTRUCTION, it));
      s = statement(structured_stmtt::WHILE, test, negate(test->guard));
      s.body = range(std::next(test), loop.latch);
    }
    else
    {
      s.kind =
        do_while ? structured_stmtt::DO_WHILE : structured_stmtt::FOREVER;
      if (do_while)
        s.condition = loop.latch->guard;
      s.body = range(loop.head, loop.latch);
    }
    open_loops.pop_back();
    s.body.push_back(statement(structured_stmtt::ANCHOR, loop.latch));
    s.loop = loop.number;
    s.head = loop.head;
    s.latch = loop.latch;
    out.push_back(std::move(s));
    return after;
  }

  std::optional<structured_stmtt::kindt> loop_jump(targett target) const
  {
    if (open_loops.empty())
      return std::nullopt;
    if (target == open_loops.back().after)
      return structured_stmtt::BREAK;
    if (target == open_loops.back().next_iteration)
      return structured_stmtt::CONTINUE;
    return std::nullopt;
  }

  targett recover_goto(targett it, targett last, stmtst &out)
  {
    if (it->targets.size() == 1)
    {
      const targett target = it->targets.front();
      if (const auto jump = loop_jump(target))
      {
        out.push_back(statement(*jump, it, it->guard));
        return std::next(it);
      }
      if (
        !is_true(it->guard) && !target->is_end_function() &&
        at(target) > at(it) && at(target) <= at(last) &&
        can_enclose(std::next(it), target))
        return recover_if(it, target, last, out);
    }
    structured_stmtt s = statement(structured_stmtt::GOTO, it, it->guard);
    s.targets.assign(it->targets.begin(), it->targets.end());
    out.push_back(std::move(s));
    return std::next(it);
  }

  /// Where the else part ends, or @p target if there is none.
  targett else_end(targett it, targett target, targett last) const
  {
    const targett jump = std::prev(target);
    if (
      jump == it || !jump->is_goto() || jump->targets.size() != 1 ||
      !is_true(jump->guard))
      return target;
    const targett end = jump->targets.front();
    const bool fits = at(end) > at(target) && at(end) <= at(last) &&
                      !end->is_end_function() && !loop_jump(end) &&
                      can_enclose(target, end);
    return fits ? end : target;
  }

  targett recover_if(targett it, targett target, targett last, stmtst &out)
  {
    const nestingt nesting(depth);
    structured_stmtt s = statement(structured_stmtt::IF, it, negate(it->guard));
    const targett end = else_end(it, target, last);
    if (end != target)
    {
      const targett jump = std::prev(target);
      s.body = range(std::next(it), jump);
      s.body.push_back(statement(structured_stmtt::ANCHOR, jump));
      s.otherwise = range(target, end);
    }
    else
      s.body = range(std::next(it), target);
    out.push_back(std::move(s));
    return end;
  }
};
} // namespace

std::vector<structured_stmtt>
recover_structure(const goto_programt &program, const enclosablet &enclosable)
{
  return structure_recoveryt(program, enclosable).recover();
}
