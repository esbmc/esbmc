#include <goto-programs/write_footprint.h>
#include <goto-programs/goto_functions.h>
#include <util/config/config.h>
#include <util/expr/type_byte_size.h>
#include <util/base/prefix.h>
#include <algorithm>
#include <optional>

namespace
{
constexpr unsigned max_query_steps = 4096;
constexpr unsigned max_layout_nodes = 4096;

struct ranget
{
  BigInt lower, upper;
};

// Inclusive bounds on possible write-start byte offsets from each root object.
using regionst = std::unordered_map<expr2tc, ranget, irep2_hash>;

bool supported_layout(const type2tc &type, unsigned &remaining)
{
  if (remaining == 0)
    return false;
  --remaining;
  if (is_array_type(type))
  {
    const auto &a = to_array_type(type);
    return !a.size_is_infinite && is_constant_int2t(a.array_size) &&
           to_constant_int2t(a.array_size).value > 0 &&
           supported_layout(a.subtype, remaining);
  }
  if (is_struct_type(type))
  {
    const auto &s = to_struct_type(type);
    return std::all_of(
      s.members.begin(), s.members.end(), [&](const type2tc &m) {
        return supported_layout(m, remaining);
      });
  }
  return is_bv_type(type) || is_bool_type(type) || is_floatbv_type(type) ||
         is_pointer_type(type);
}

bool supported_layout(const type2tc &type)
{
  // Size utilities expand shared nested types, so bound the expanded layout.
  unsigned remaining = max_layout_nodes;
  return supported_layout(type, remaining);
}

std::optional<ranget> integer_range(const expr2tc &e)
{
  if (is_constant_int2t(e))
    return ranget{to_constant_int2t(e).value, to_constant_int2t(e).value};
  if (!is_signedbv_type(e->type) && !is_unsignedbv_type(e->type))
    return {};
  const unsigned width = e->type->get_width();
  if (width == 0)
    return {};
  const bool signed_type = is_signedbv_type(e->type);
  const BigInt high = (BigInt(1) << (signed_type ? width - 1 : width)) - 1;
  ranget result{signed_type ? -high - 1 : BigInt(0), high};
  if (is_typecast2t(e))
    if (
      auto source = integer_range(to_typecast2t(e).from);
      source && source->lower >= result.lower && source->upper <= result.upper)
      return source;
  return result;
}

bool merge(regionst &into, const regionst &from, unsigned &budget)
{
  if (from.size() > budget)
    return false;
  budget -= from.size();
  for (const auto &[object, range] : from)
  {
    auto [it, fresh] = into.emplace(object, range);
    if (!fresh)
    {
      it->second.lower = std::min(it->second.lower, range.lower);
      it->second.upper = std::max(it->second.upper, range.upper);
    }
  }
  return true;
}

class address_rangest
{
public:
  explicit address_rangest(const write_footprintt::sourcest &sources)
    : sources(sources),
      limit((BigInt(1) << (config.ansi_c.address_width - 1)) - 1)
  {
  }

  std::optional<regionst> write_address(const expr2tc &lhs)
  {
    budget = max_query_steps;
    auto regions = address(lhs);
    if (!regions || !supported_layout(lhs->type))
      return {};
    // A representable start does not guarantee that the complete store fits.
    const BigInt size = type_byte_size(lhs->type);
    for (const auto &[object, range] : *regions)
      if (range.upper + size - 1 > limit)
        return {};
    return regions;
  }

private:
  std::optional<regionst> address(const expr2tc &e)
  {
    if (budget == 0)
      return {};
    --budget;
    if (is_symbol2t(e))
      return regionst{{e, {0, 0}}};
    if (is_dereference2t(e))
      return pointer(to_dereference2t(e).value);
    if (is_member2t(e))
    {
      const auto &m = to_member2t(e);
      if (
        !is_struct_type(m.source_value->type) ||
        !supported_layout(m.source_value->type))
        return {};
      const BigInt bits = member_offset_bits(m.source_value->type, m.member);
      if (bits % 8 != 0)
        return {};
      return shift(address(m.source_value), {bits / 8, bits / 8}, 1);
    }
    if (is_index2t(e))
    {
      const auto &i = to_index2t(e);
      auto range = integer_range(i.index);
      if (!range || !supported_layout(e->type))
        return {};
      auto base = is_pointer_type(i.source_value->type)
                    ? pointer(i.source_value)
                    : address(i.source_value);
      return shift(std::move(base), *range, type_byte_size(e->type));
    }
    return {};
  }

  std::optional<regionst> shift(
    std::optional<regionst> regions,
    const ranget &offset,
    const BigInt &scale)
  {
    if (!regions || scale <= 0)
      return {};
    for (auto &[object, range] : *regions)
    {
      range.lower += offset.lower * scale;
      range.upper += offset.upper * scale;
      // Pointer offsets use target-width arithmetic, including on 32-bit.
      if (range.lower < -limit - 1 || range.upper > limit)
        return {};
    }
    return regions;
  }

  std::optional<regionst> pointer(const expr2tc &e)
  {
    if (budget == 0)
      return {};
    --budget;
    if (is_address_of2t(e))
      return address(to_address_of2t(e).ptr_obj);
    if (is_array_type(e->type))
      return address(e);
    if (!is_pointer_type(e->type))
      return {};
    if (is_typecast2t(e))
    {
      const auto &cast = to_typecast2t(e);
      if (cast.base_to_derived || !cast.derived_to_base.empty())
        return {};
      return pointer(cast.from);
    }
    if (is_if2t(e))
    {
      auto a = pointer(to_if2t(e).true_value);
      auto b = pointer(to_if2t(e).false_value);
      if (!a || !b)
        return {};
      if (!merge(*a, *b, budget))
        return {};
      return a;
    }
    if (is_add2t(e) || is_sub2t(e))
      return arithmetic(e);
    if (!is_symbol2t(e))
      return {};
    return symbol(e);
  }

  std::optional<regionst> arithmetic(const expr2tc &e)
  {
    expr2tc base = *e->get_sub_expr(0);
    expr2tc offset = *e->get_sub_expr(1);
    if (is_add2t(e) && is_pointer_type(offset->type))
      std::swap(base, offset);
    if (!is_pointer_type(base->type))
      return {};
    const type2tc subtype = to_pointer_type(base->type).subtype;
    auto range = integer_range(offset);
    if (!range || !supported_layout(subtype))
      return {};
    if (is_sub2t(e))
      *range = {-range->upper, -range->lower};
    return shift(pointer(base), *range, type_byte_size(subtype));
  }

  std::optional<regionst> symbol(const expr2tc &e)
  {
    const irep_idt &name = to_symbol2t(e).thename;
    if (auto it = cache.find(name); it != cache.end())
      return it->second;
    if (!visiting.insert(name).second)
      return {};
    const auto values = sources(e);
    std::optional<regionst> result = regionst{};
    if (values.empty())
      result.reset();
    for (const expr2tc &value : values)
    {
      auto part = pointer(value);
      if (!part || !merge(*result, *part, budget))
      {
        result.reset();
        break;
      }
    }
    visiting.erase(name);
    cache.emplace(name, result);
    return result;
  }

  const write_footprintt::sourcest &sources;
  const BigInt limit;
  unsigned budget = max_query_steps;
  std::unordered_map<irep_idt, std::optional<regionst>, irep_id_hash> cache;
  std::unordered_set<irep_idt, irep_id_hash> visiting;
};

class footprint_collectort
{
public:
  footprint_collectort(
    value_setst &points_to,
    const write_footprintt::sourcest &sources)
    : points_to(points_to), ranges(sources)
  {
  }

  void run(
    const goto_functionst &functions,
    const std::unordered_set<irep_idt, irep_id_hash> &reachable,
    const write_footprintt::in_loopt &in_loop)
  {
    if (spawns_threads(functions))
    {
      unknown = true;
      return;
    }
    for (const auto &[name, function] : functions.function_map)
      if (function.body_available && reachable.count(name))
        for (auto i = function.body.instructions.begin();
             i != function.body.instructions.end();
             ++i)
        {
          if (!in_loop(name, i->location_number))
            continue;
          // Writes such as printf's %n can also occur inside an assignment RHS.
          unsigned remaining = max_query_steps;
          record_implicit_writes(i, i->code, remaining);
          record_implicit_writes(i, i->guard, remaining);
          if (unknown)
            return;
          if (i->is_assign())
            record_store(i, to_code_assign2t(i->code).target, false);
          else if (i->is_function_call())
            record_call(functions, i);
          else if (
            i->is_other() && !is_code_expression2t(i->code) &&
            !is_code_printf2t(i->code))
            unknown = true;
          if (unknown)
            return;
        }
  }

  std::unordered_map<irep_idt, BigInt, irep_id_hash> lower_bounds;
  bool unknown = false;

private:
  void record_implicit_writes(
    goto_programt::const_targett i,
    const expr2tc &e,
    unsigned &remaining)
  {
    if (unknown || is_nil_expr(e))
      return;
    if (remaining == 0)
    {
      unknown = true;
      return;
    }
    --remaining;
    if (
      is_sideeffect2t(e) &&
      to_sideeffect2t(e).kind != sideeffect2t::allockind::nondet)
    {
      unknown = true;
      return;
    }
    if (is_code_printf2t(e))
    {
      const auto &output = to_code_printf2t(e);
      switch (output.kind)
      {
      case printf_kindt::PRINTF:
      case printf_kindt::FPRINTF:
      case printf_kindt::DPRINTF:
      case printf_kindt::SPRINTF:
      case printf_kindt::SNPRINTF:
      case printf_kindt::ASPRINTF:
        record_pointer_arguments(i, output.operands);
        break;
      default:
        // A va_list can hide %n targets absent from the direct operands.
        unknown = true;
        return;
      }
    }
    e->foreach_operand(
      [&](const expr2tc &op) { record_implicit_writes(i, op, remaining); });
  }

  void record_pointer_arguments(
    goto_programt::const_targett i,
    const std::vector<expr2tc> &arguments)
  {
    for (const auto &arg : arguments)
      if (!is_nil_expr(arg) && is_pointer_type(arg->type))
        record_store(
          i, dereference2tc(to_pointer_type(arg->type).subtype, arg), true);
  }

  std::vector<expr2tc>
  targets(goto_programt::const_targett i, const expr2tc &lhs)
  {
    value_setst::valuest objects;
    points_to.get_values(i, address_of2tc(lhs->type, lhs), objects);
    if (objects.empty())
      unknown = true;
    std::vector<expr2tc> result;
    for (const auto &v : objects)
    {
      if (!is_object_descriptor2t(v))
      {
        unknown = true;
        continue;
      }
      const expr2tc &object = to_object_descriptor2t(v).object;
      if (is_symbol2t(object))
      {
        if (is_struct_type(object->type))
          result.push_back(object);
        else
        {
          // A later query may resolve this identifier to a struct type.
          lower_bounds[to_symbol2t(object).thename] = 0;
        }
      }
      else if (
        !is_dynamic_object2t(object) && !is_constant_string2t(object) &&
        !is_null_object2t(object))
        unknown = true;
    }
    return result;
  }

  void record_store(
    goto_programt::const_targett i,
    const expr2tc &lhs,
    bool whole_object)
  {
    if (is_nil_expr(lhs))
      return;
    // Syntactic origins only refine bounds; the alias oracle decides targets.
    const auto objects = targets(i, lhs);
    if (objects.empty())
      return;
    auto regions =
      whole_object ? std::optional<regionst>{} : ranges.write_address(lhs);
    for (const auto &object : objects)
    {
      BigInt lower = 0;
      if (regions)
        if (auto it = regions->find(object); it != regions->end())
          lower = std::max(BigInt(0), it->second.lower);
      auto [it, fresh] =
        lower_bounds.emplace(to_symbol2t(object).thename, lower);
      if (!fresh)
        it->second = std::min(it->second, lower);
    }
  }

  void
  record_call(const goto_functionst &functions, goto_programt::const_targett i)
  {
    const auto &call = to_code_function_call2t(i->code);
    record_store(i, call.ret, false);
    const auto callee =
      is_symbol2t(call.function)
        ? functions.function_map.find(to_symbol2t(call.function).thename)
        : functions.function_map.end();
    // Intrinsics may write through arguments independently of their GOTO body.
    if (
      callee != functions.function_map.end() && callee->second.body_available &&
      !callee->second.body.hide &&
      !has_prefix(
        to_symbol2t(call.function).thename.as_string(), "c:@F@__ESBMC"))
      return;
    record_pointer_arguments(i, call.operands);
  }

  value_setst &points_to;
  address_rangest ranges;
};
} // namespace

write_footprintt::write_footprintt(
  const goto_functionst &functions,
  value_setst &points_to,
  const std::unordered_set<irep_idt, irep_id_hash> &reachable,
  const in_loopt &in_loop,
  const sourcest &sources)
{
  footprint_collectort collector(points_to, sources);
  collector.run(functions, reachable, in_loop);
  // One unresolved effect can invalidate every retained object prefix.
  if (!collector.unknown)
    lower_bounds = std::move(collector.lower_bounds);
}

const std::vector<expr2tc> *
write_footprintt::partial_writes(const expr2tc &object) const
{
  if (!is_symbol2t(object) || !is_struct_type(object->type))
    return nullptr;
  if (auto it = member_targets.find(object); it != member_targets.end())
    return &it->second;
  const auto it = lower_bounds.find(to_symbol2t(object).thename);
  if (
    it == lower_bounds.end() || it->second <= 0 ||
    !supported_layout(object->type))
    return nullptr;
  auto &fields = member_targets[object];
  const auto &s = to_struct_type(object->type);
  BigInt end = 0;
  // Havoc whole fields overlapping the suffix; never generate byte updates.
  for (size_t i = 0; i < s.members.size(); ++i)
  {
    end += type_byte_size_bits(s.members[i]);
    if (end > it->second * 8)
      fields.push_back(member2tc(s.members[i], object, s.member_names[i]));
  }
  return &fields;
}
