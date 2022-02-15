/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_symex_state.h>
#include <goto-symex/symex_target_equation.h>
#include <langapi/language_util.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/std_expr.h>
#include <util/message/default_message.h>

void symex_target_equationt::debug_print_step(const SSA_stept &step) const
{
  default_message msg;
  std::ostringstream oss;
  step.output(ns, oss, msg);
  msg.debug(oss.str());
}

void symex_target_equationt::assignment(
  const expr2tc &guard,
  const expr2tc &lhs,
  const expr2tc &original_lhs,
  const expr2tc &rhs,
  const expr2tc &original_rhs,
  const sourcet &source,
  std::vector<stack_framet> stack_trace,
  const bool hidden,
  unsigned loop_number)
{
  assert(!is_nil_expr(lhs));

  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = lhs;
  SSA_step.original_lhs = original_lhs;
  SSA_step.original_rhs = original_rhs;
  SSA_step.rhs = rhs;
  SSA_step.hidden = hidden;
  SSA_step.cond = equality2tc(lhs, rhs);
  SSA_step.type = goto_trace_stept::ASSIGNMENT;
  SSA_step.source = source;
  SSA_step.stack_trace = stack_trace;
  SSA_step.loop_number = loop_number;

  if(debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::output(
  const expr2tc &guard,
  const sourcet &source,
  const std::string &fmt,
  const std::list<expr2tc> &args)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.type = goto_trace_stept::OUTPUT;
  SSA_step.source = source;
  SSA_step.output_args = args;
  SSA_step.format_string = fmt;

  if(debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::assumption(
  const expr2tc &guard,
  const expr2tc &cond,
  const sourcet &source,
  unsigned loop_number)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type = goto_trace_stept::ASSUME;
  SSA_step.source = source;
  SSA_step.loop_number = loop_number;

  if(debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::assertion(
  const expr2tc &guard,
  const expr2tc &cond,
  const std::string &msg,
  std::vector<stack_framet> stack_trace,
  const sourcet &source,
  unsigned loop_number)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type = goto_trace_stept::ASSERT;
  SSA_step.source = source;
  SSA_step.comment = msg;
  SSA_step.stack_trace = stack_trace;
  SSA_step.loop_number = loop_number;

  if(debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::renumber(
  const expr2tc &guard,
  const expr2tc &symbol,
  const expr2tc &size,
  const sourcet &source)
{
  assert(is_symbol2t(symbol));
  assert(is_bv_type(size));
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = symbol;
  SSA_step.rhs = size;
  SSA_step.type = goto_trace_stept::RENUMBER;
  SSA_step.source = source;

  if(debug_print)
    debug_print_step(SSA_step);
}

void symex_target_equationt::convert(smt_convt &smt_conv)
{
  smt_convt::ast_vec assertions;
  smt_astt assumpt_ast = smt_conv.convert_ast(gen_true_expr());

  for(auto &SSA_step : SSA_steps)
    convert_internal_step(smt_conv, assumpt_ast, assertions, SSA_step);

  if(!assertions.empty())
    smt_conv.assert_ast(
      smt_conv.make_n_ary(&smt_conv, &smt_convt::mk_or, assertions));
}

static void flatten_to_bytes(const expr2tc &new_expr, std::vector<expr2tc> &bytes)
{
  // Awkwardly, this array literal might not be completely fixed-value, if
  // encoded in the middle of a function body or something that refers to other
  // variables.
  if(is_array_type(new_expr))
  {
    // Assume only fixed-size arrays (because you can't have variable size
    // members of unions).
    const array_type2t &arraytype = to_array_type(new_expr->type);
    assert(
      !arraytype.size_is_infinite && !is_nil_expr(arraytype.array_size) &&
      is_constant_int2t(arraytype.array_size) &&
      "Can't flatten array in union literal with unbounded size");

    // Iterate over each field and flatten to bytes
    const constant_int2t &intref = to_constant_int2t(arraytype.array_size);
    for(unsigned int i = 0; i < intref.value.to_uint64(); i++)
    {
      index2tc idx(arraytype.subtype, new_expr, gen_ulong(i));
      flatten_to_bytes(idx, bytes);
    }
  }
  else if(is_struct_type(new_expr))
  {
    // Iterate over each field.
    const struct_type2t &structtype = to_struct_type(new_expr->type);
    BigInt member_offset_bits = 0;
    bool been_inside_bits_region = false;
    BigInt bitfields_first_byte = 0;
    for(unsigned long i = 0; i < structtype.members.size(); i++)
    {
      BigInt member_size_bits = type_byte_size_bits(structtype.members[i]);
      // If this member is a bitfield, extract everything as it is until
      // the next member aligned to a byte as all such members can only
      // be of scalar type
      if(member_size_bits % 8 != 0 || member_offset_bits % 8 != 0)
      {
        // This is the first bit-field within the region.
        // So we update its first byte
        if(!been_inside_bits_region)
        {
          bitfields_first_byte = member_offset_bits / 8;
          been_inside_bits_region = true;
        }
      }
      else
      {
        // This means that we just came out of the region comprised by
        // bit-fields and we need to extract this region as it is
        // before we try extract current member
        if(been_inside_bits_region)
        {
          bool is_big_endian =
            config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN;
          for(unsigned long j = bitfields_first_byte.to_uint64();
              j < (member_offset_bits / 8).to_uint64();
              j++)
          {
            byte_extract2tc struct_byte(
              get_uint8_type(), new_expr, gen_ulong(j), is_big_endian);
            bytes.push_back(struct_byte);
          }
          been_inside_bits_region = false;
        }
        // Now we can flatten to bytes this member as most likely it is not a bit-field.
        // And even if it is a bit-field it is aligned to a byte and its
        // size
        member2tc memb(
          structtype.members[i], new_expr, structtype.member_names[i]);
        flatten_to_bytes(memb, bytes);
      }
      member_offset_bits += member_size_bits;
    }
    // This means that the struct ended on a bitfield.
    // Hence, we need to do some final extractions.
    if(been_inside_bits_region)
    {
      bool is_big_endian =
        config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN;
      for(unsigned long j = bitfields_first_byte.to_uint64();
          j < (member_offset_bits / 8).to_uint64();
          j++)
      {
        byte_extract2tc struct_byte(
          get_uint8_type(), new_expr, gen_ulong(j), is_big_endian);
        bytes.push_back(struct_byte);
      }
    }
  }
  else if(is_union_type(new_expr))
  {
    // This is an expression that evaluates to a union -- probably a symbol
    // name. It can't be a union literal, because that would have been
    // recursively flattened. In this circumstance we are *not* required to
    // actually perform any flattening, because something else in the union
    // transformation should have transformed it to a byte array. Simply take
    // the address (it has to have storage), cast to byte array, and index.
    BigInt size = type_byte_size(new_expr->type);
    address_of2tc addrof(new_expr->type, new_expr);
    type2tc byteptr(new pointer_type2t(get_uint8_type()));
    typecast2tc cast(byteptr, addrof);

    // Produce N bytes
    for(unsigned int i = 0; i < size.to_uint64(); i++)
    {
      index2tc idx(get_uint8_type(), cast, gen_ulong(i));
      flatten_to_bytes(idx, bytes);
    }
  }
  else if(is_number_type(new_expr) || is_pointer_type(new_expr))
  {
    BigInt size = type_byte_size(new_expr->type);

    bool is_big_endian =
      config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN;
    for(unsigned int i = 0; i < size.to_uint64(); i++)
    {
      byte_extract2tc ext(
        get_uint8_type(), new_expr, gen_ulong(i), is_big_endian);
      bytes.push_back(ext);
    }
  }
  else
  {
    assert(
      0 && fmt::format(
             "Unrecognized type {}  when flattening union literal",
             get_type_id(*new_expr->type))
             .c_str());
  }
}

static constant_array2tc flatten_union(const constant_union2t &expr)
{
  const type2tc &type = expr.type;
  BigInt full_size = type_byte_size(type);

  // Union literals should have zero/one field.
  assert(
    expr.datatype_members.size() < 2 && "Union literal with more than one field");

  // Cannot have unbounded size; flatten to an array of bytes.
  std::vector<expr2tc> byte_array;

  if(expr.datatype_members.size() == 1)
    flatten_to_bytes(expr.datatype_members[0], byte_array);

  // Potentially extend this array further if this literal is smaller than
  // the overall size of the union.
  expr2tc abyte = gen_zero(get_uint8_type());
  while(byte_array.size() < full_size.to_uint64())
    byte_array.push_back(abyte);

  expr2tc size = gen_ulong(byte_array.size());
  type2tc arraytype(new array_type2t(get_uint8_type(), size, false));
  return constant_array2tc(arraytype, byte_array);
}

static expr2tc flatten_unions(expr2tc expr);

static expr2tc flatten_with(const with2t &with)
{
  assert(with.type == with.source_value->type);
  auto *u = static_cast<const struct_union_data *>(
    dynamic_cast<const union_type2t *>(with.source_value->type.get()));
  // BigInt bits = type_byte_size_bits(with.type);
  assert(is_constant_string2t(with.update_field));
  const irep_idt &field = to_constant_string2t(with.update_field).value;
  assert(member_offset_bits(with.type, field) == 0);
  unsigned c = u->get_component_number(field);
  const type2tc &member_type = u->members[c];
  expr2tc flattened_source = flatten_unions(with.source_value /*, member_type */);
  assert(is_array_type(flattened_source->type));
  assert(to_array_type(flattened_source->type).subtype == get_uint8_type());
  expr2tc flattened_value = flatten_unions(with.update_value);
  BigInt bits = type_byte_size_bits(member_type);
  if(bits % 8 == 0)
  {
    BigInt bytes = bits / 8;
    assert(bytes.is_uint64());
    uint64_t bytes64 = bytes.to_uint64();
    assert(bytes64 <= UINT_MAX);
    std::vector<expr2tc> value_bytes;
    flatten_to_bytes(flattened_value, value_bytes);
    assert(bytes64 <= value_bytes.size());
    expr2tc res;
    if(is_constant_expr(flattened_source))
    {
      constant_array2tc tgt(flattened_source);
      for(size_t i = 0; i < bytes64; i++)
        tgt->datatype_members[i] = value_bytes[i];
      res = tgt;
    }
    else
    {
      res = flattened_source;
      for(size_t i = 0; i < bytes64; i++)
        res = with2tc(res->type, res, gen_ulong(i), value_bytes[i]);
    }
    return res;
  }
  else
    assert(0 && "unimplemented");
}

static expr2tc flatten_unions(expr2tc expr)
{
  struct
  {
    void operator()(expr2tc &e) const
    {
      if(is_union_type(e))
      {
        if(is_with2t(e))
        {
          e = flatten_with(to_with2t(e));
        }
        else if(is_symbol2t(e))
        {
          BigInt bytes = type_byte_size(e->type);
          assert(bytes.is_uint64());
          uint64_t bytes64 = bytes.to_uint64();
          assert(bytes64 <= ULONG_MAX);
          e->type = array_type2tc(get_uint8_type(), gen_ulong(bytes64), false);
        }
        else if(is_constant_expr(e))
        {
          assert(is_constant_union2t(e));
          e = flatten_union(to_constant_union2t(e));
        }
        else
          assert(0);
      }
      else if(is_member2t(e) && is_union_type(to_member2t(e).source_value))
      {
        const member2t &member = to_member2t(e);
        auto *u = static_cast<const struct_union_data *>(
          dynamic_cast<const union_type2t *>(member.source_value->type.get()));
        unsigned c = u->get_component_number(member.member);
        assert(
          member_offset_bits(member.source_value->type, member.member) == 0);
        const type2tc &member_type = u->members[c];
        expr2tc flattened_source = flatten_unions(member.source_value);
        assert(is_array_type(flattened_source));
        BigInt bits = type_byte_size_bits(member_type);
        if(bits % 8 == 0)
        {
          BigInt bytes = bits / 8;
          assert(bytes.is_uint64());
          uint64_t bytes64 = bytes.to_uint64();
          assert(bytes64 <= ULONG_MAX);
          guardt guard;
          e = dereferencet::stitch_together_from_byte_array(
            member_type, bytes64, flattened_source, gen_ulong(0), guard);
        }
        else
          assert(0);
      }
      else
        e->Foreach_operand(*this);
    }
  } do_flatten;
  do_flatten(expr);
  return expr;
}

void symex_target_equationt::convert_internal_step(
  smt_convt &smt_conv,
  smt_astt &assumpt_ast,
  smt_convt::ast_vec &assertions,
  SSA_stept &step)
{
  static unsigned output_count = 0; // Temporary hack; should become scoped.
  smt_astt true_val = smt_conv.convert_ast(gen_true_expr());
  smt_astt false_val = smt_conv.convert_ast(gen_false_expr());

  if(step.ignore)
  {
    step.cond_ast = true_val;
    step.guard_ast = false_val;
    return;
  }

  if(ssa_trace)
  {
    std::ostringstream oss;
    step.output(ns, oss, msg);
    msg.status(oss.str());
  }

  step.guard_ast = smt_conv.convert_ast(flatten_unions(step.guard));

  if(step.is_assume() || step.is_assert())
  {
    expr2tc tmp(flatten_unions(step.cond));
    step.cond_ast = smt_conv.convert_ast(tmp);

    if(ssa_smt_trace)
    {
      step.cond_ast->dump();
    }
  }
  else if(step.is_assignment())
  {
    smt_astt assign = smt_conv.convert_assign(flatten_unions(step.cond));
    if(ssa_smt_trace)
    {
      assign->dump();
    }
  }
  else if(step.is_output())
  {
    for(std::list<expr2tc>::const_iterator o_it = step.output_args.begin();
        o_it != step.output_args.end();
        o_it++)
    {
      const expr2tc &tmp = *o_it;
      if(is_constant_expr(tmp) || is_constant_string2t(tmp))
        step.converted_output_args.push_back(tmp);
      else
      {
        symbol2tc sym(tmp->type, "symex::output::" + i2string(output_count++));
        equality2tc eq(sym, tmp);
        smt_conv.set_to(eq, true);
        step.converted_output_args.push_back(sym);
      }
    }
  }
  else if(step.is_renumber())
  {
    smt_conv.renumber_symbol_address(step.guard, step.lhs, step.rhs);
  }
  else if(!step.is_skip())
  {
    assert(0 && "Unexpected SSA step type in conversion");
  }

  if(step.is_assert())
  {
    step.cond_ast = smt_conv.imply_ast(assumpt_ast, step.cond_ast);
    assertions.push_back(smt_conv.invert_ast(step.cond_ast));
  }
  else if(step.is_assume())
  {
    smt_convt::ast_vec v;
    v.push_back(assumpt_ast);
    v.push_back(step.cond_ast);
    assumpt_ast = smt_conv.make_n_ary(&smt_conv, &smt_convt::mk_and, v);
  }
}

void symex_target_equationt::output(std::ostream &out) const
{
  for(const auto &SSA_step : SSA_steps)
  {
    SSA_step.output(ns, out, msg);
    out << "--------------"
        << "\n";
  }
}

void symex_target_equationt::short_output(std::ostream &out, bool show_ignored)
  const
{
  for(const auto &SSA_step : SSA_steps)
  {
    SSA_step.short_output(ns, out, msg, show_ignored);
  }
}

void symex_target_equationt::SSA_stept::dump() const
{
  default_message msg;
  std::ostringstream oss;
  output(*migrate_namespace_lookup, oss, msg);
  msg.debug(oss.str());
}

void symex_target_equationt::SSA_stept::output(
  const namespacet &ns,
  std::ostream &out,
  const messaget &msg) const
{
  if(source.is_set)
  {
    out << "Thread " << source.thread_nr;

    if(source.pc->location.is_not_nil())
      out << " " << source.pc->location << "\n";
    else
      out << "\n";
  }

  switch(type)
  {
  case goto_trace_stept::ASSERT:
    out << "ASSERT"
        << "\n";
    break;
  case goto_trace_stept::ASSUME:
    out << "ASSUME"
        << "\n";
    break;
  case goto_trace_stept::OUTPUT:
    out << "OUTPUT"
        << "\n";
    break;

  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT (";
    out << (hidden ? "HIDDEN" : "") << ")\n";
    break;

  default:
    assert(false);
  }

  if(is_assert() || is_assume() || is_assignment())
    out << from_expr(ns, "", migrate_expr_back(cond), msg) << "\n";

  if(is_assert())
    out << comment << "\n";

  if(config.options.get_bool_option("ssa-guards"))
    out << "Guard: " << from_expr(ns, "", migrate_expr_back(guard), msg)
        << "\n";
}

void symex_target_equationt::SSA_stept::short_output(
  const namespacet &ns,
  std::ostream &out,
  const messaget &msg,
  bool show_ignored) const
{
  if((is_assignment() || is_assert() || is_assume()) && show_ignored == ignore)
  {
    out << from_expr(ns, "", cond, msg) << "\n";
  }
  else if(is_renumber())
  {
    out << "renumber: " << from_expr(ns, "", lhs, msg) << "\n";
  }
}

void symex_target_equationt::push_ctx()
{
}

void symex_target_equationt::pop_ctx()
{
}

std::ostream &
operator<<(std::ostream &out, const symex_target_equationt &equation)
{
  equation.output(out);
  return out;
}

void symex_target_equationt::check_for_duplicate_assigns() const
{
  std::map<std::string, unsigned int> countmap;
  unsigned int i = 0;

  for(const auto &SSA_step : SSA_steps)
  {
    i++;
    if(!SSA_step.is_assignment())
      continue;

    const equality2t &ref = to_equality2t(SSA_step.cond);
    const symbol2t &sym = to_symbol2t(ref.side_1);
    countmap[sym.get_symbol_name()]++;
  }

  for(std::map<std::string, unsigned int>::const_iterator it = countmap.begin();
      it != countmap.end();
      it++)
  {
    if(it->second != 1)
    {
      msg.status(
        fmt::format("Symbol \"{}\" appears {} times", it->first, it->second));
    }
  }

  msg.status(fmt::format("Checked {} insns", i));
}

unsigned int symex_target_equationt::clear_assertions()
{
  unsigned int num_asserts = 0;

  for(SSA_stepst::iterator it = SSA_steps.begin(); it != SSA_steps.end(); it++)
  {
    if(it->type == goto_trace_stept::ASSERT)
    {
      SSA_stepst::iterator it2 = it;
      it--;
      SSA_steps.erase(it2);
      num_asserts++;
    }
  }

  return num_asserts;
}

runtime_encoded_equationt::runtime_encoded_equationt(
  const namespacet &_ns,
  smt_convt &_conv,
  const messaget &msg)
  : symex_target_equationt(_ns, msg), conv(_conv)
{
  assert_vec_list.emplace_back();
  assumpt_chain.push_back(conv.convert_ast(gen_true_expr()));
  cvt_progress = SSA_steps.end();
}

void runtime_encoded_equationt::flush_latest_instructions()
{
  if(SSA_steps.size() == 0)
    return;

  SSA_stepst::iterator run_it = cvt_progress;
  // Scenarios:
  // * We're at the start of running, in which case cvt_progress == end
  // * We're in the middle, but nothing is left to push, so run_it + 1 == end
  // * We're in the middle, and there's more to convert.
  if(run_it == SSA_steps.end())
  {
    run_it = SSA_steps.begin();
  }
  else
  {
    run_it++;
    if(run_it == SSA_steps.end())
    {
      // There is in fact, nothing to do
      return;
    }

    // Just roll on
  }

  // Now iterate from the start insn to convert, to the end of the list.
  for(; run_it != SSA_steps.end(); ++run_it)
    convert_internal_step(
      conv, assumpt_chain.back(), assert_vec_list.back(), *run_it);

  run_it--;
  cvt_progress = run_it;
}

void runtime_encoded_equationt::push_ctx()
{
  flush_latest_instructions();

  // And push everything back.
  assumpt_chain.push_back(assumpt_chain.back());
  assert_vec_list.push_back(assert_vec_list.back());
  scoped_end_points.push_back(cvt_progress);
  conv.push_ctx();
}

void runtime_encoded_equationt::pop_ctx()
{
  SSA_stepst::iterator it = scoped_end_points.back();
  cvt_progress = it;

  if(SSA_steps.size() != 0)
    ++it;

  SSA_steps.erase(it, SSA_steps.end());

  conv.pop_ctx();
  scoped_end_points.pop_back();
  assert_vec_list.pop_back();
  assumpt_chain.pop_back();
}

void runtime_encoded_equationt::convert(smt_convt &smt_conv)
{
  // Don't actually convert. We've already done most of the conversion by now
  // (probably), instead flush all unconverted instructions. We don't push
  // a context, because a) where do we unpop it, but b) we're never going to
  // build anything on top of this, so there's no gain by pushing it.
  flush_latest_instructions();

  // Finally, we also want to assert the set of assertions.
  if(!assert_vec_list.back().empty())
    smt_conv.assert_ast(smt_conv.make_n_ary(
      &smt_conv, &smt_convt::mk_or, assert_vec_list.back()));
}

std::shared_ptr<symex_targett> runtime_encoded_equationt::clone() const
{
  // Only permit cloning at the start of a run - there should never be any data
  // in this formula when it happens. Cloning needs to be supported so that a
  // reachability_treet can take a template equation and clone it ever time it
  // sets up a new exploration.
  assert(
    SSA_steps.size() == 0 &&
    "runtime_encoded_equationt shouldn't be "
    "cloned when it contains data");
  auto nthis = std::shared_ptr<runtime_encoded_equationt>(
    new runtime_encoded_equationt(*this));
  nthis->cvt_progress = nthis->SSA_steps.end();
  return nthis;
}

tvt runtime_encoded_equationt::ask_solver_question(const expr2tc &question)
{
  tvt final_res;

  // So - we have a formula, we want to work out whether it's true, false, or
  // unknown. Before doing anything, first push a context, as we'll need to
  // wipe some state afterwards.
  push_ctx();

  // Convert the question (must be a bool).
  assert(is_bool_type(question));
  smt_astt q = conv.convert_ast(question);

  // The proposition also needs to be guarded with the in-program assumptions,
  // which are not necessarily going to be part of the state guard.
  conv.assert_ast(assumpt_chain.back());

  // Now, how to ask the question? Unfortunately the clever solver stuff won't
  // negate the condition, it'll only give us a handle to it that it negates
  // when we access. So, we have to make an assertion, check it, pop it, then
  // check another.
  // Those assertions are just is-the-prop-true, is-the-prop-false. Valid
  // results are true, false, both.
  push_ctx();
  conv.assert_ast(q);
  smt_convt::resultt res1 = conv.dec_solve();
  pop_ctx();
  push_ctx();
  conv.assert_ast(conv.invert_ast(q));
  smt_convt::resultt res2 = conv.dec_solve();
  pop_ctx();

  // So; which result?
  if(
    res1 == smt_convt::P_ERROR || res1 == smt_convt::P_SMTLIB ||
    res2 == smt_convt::P_ERROR || res2 == smt_convt::P_SMTLIB)
  {
    msg.error("Solver returned error while asking question");
    abort();
  }
  else if(res1 == smt_convt::P_SATISFIABLE && res2 == smt_convt::P_SATISFIABLE)
  {
    // Both ways are satisfiable; result is unknown.
    final_res = tvt(tvt::TV_UNKNOWN);
  }
  else if(
    res1 == smt_convt::P_SATISFIABLE && res2 == smt_convt::P_UNSATISFIABLE)
  {
    // Truth of question is satisfiable; other not; so we're true.
    final_res = tvt(tvt::TV_TRUE);
  }
  else if(
    res1 == smt_convt::P_UNSATISFIABLE && res2 == smt_convt::P_SATISFIABLE)
  {
    // Truth is unsat, false is sat, proposition is false
    final_res = tvt(tvt::TV_FALSE);
  }
  else
  {
    pop_ctx();
    throw dual_unsat_exception();
  }

  // We have our result; pop off the questions / formula we've asked.
  pop_ctx();

  return final_res;
}
