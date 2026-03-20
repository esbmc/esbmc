#include <python-frontend/cmath_lowering_policy.h>

namespace cmath_lowering_policy
{

static void
measure_recursive(const nlohmann::json &node, size_t depth, expr_cost &cost)
{
  ++cost.node_count;
  if (depth > cost.depth)
    cost.depth = depth;

  if (!node.is_object())
    return;

  if (node.contains("_type"))
  {
    const auto &type = node["_type"];
    if (type == "Call")
      ++cost.call_count;
    else if (type == "BinOp")
      ++cost.binop_count;
  }

  for (const auto &[key, child] : node.items())
  {
    if (key == "_type")
      continue;
    if (child.is_object())
      measure_recursive(child, depth + 1, cost);
    else if (child.is_array())
    {
      for (const auto &elem : child)
      {
        if (elem.is_object())
          measure_recursive(elem, depth + 1, cost);
      }
    }
  }
}

expr_cost measure(const nlohmann::json &arg)
{
  expr_cost cost;
  measure_recursive(arg, 0, cost);
  return cost;
}

bool within_budget(const nlohmann::json &arg, const budget &b)
{
  const expr_cost cost = measure(arg);
  return cost.node_count <= b.max_node_count && cost.depth <= b.max_depth &&
         cost.call_count <= b.max_call_count &&
         cost.binop_count <= b.max_binop_count;
}

} // namespace cmath_lowering_policy
