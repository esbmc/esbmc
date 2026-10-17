/// \file solidity_convert_inheritance.cpp
/// \brief Contract inheritance handling for the Solidity frontend.
///
/// Implements Solidity's C3-linearized contract inheritance model. Merges
/// base contract members (state variables, functions, modifiers) into the
/// derived contract's AST, handles virtual function override resolution,
/// and adds inheritance labels to track which contract originally defined
/// each member.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/arith/mp_arith.h>
#include <util/irep/std_expr.h>
#include <util/message/message.h>
#include <fstream>

void solidity_convertert::add_inherit_label(
  nlohmann::json &node,
  const std::string &cname)
{
  // Add or update the "is_inherited" label in the current node
  if (node.is_object() && node.contains("id"))
  {
    node["current_contract"] = cname;
    node["is_inherited"] = true;
  }

  // Traverse through all sub-nodes
  for (auto &sub_node : node)
  {
    if (sub_node.is_object() && sub_node.contains("id"))
    {
      sub_node["current_contract"] = cname;
      sub_node["is_inherited"] = true;
    }

    if (sub_node.is_object() || sub_node.is_array())
      add_inherit_label(sub_node, cname);
  }
}

/*
  prefix:
    c_: current contract, we need to merged the inherited contract nodes to it
    i_: inherited contract
*/
void solidity_convertert::merge_inheritance_ast(
  const std::string &c_name,
  nlohmann::json &c_node,
  std::set<std::string> &merged_list)
{
  log_debug("solidity", "@@@ Merging AST for contract {}", c_name);
  // we have merged this contract
  if (merged_list.count(c_name) > 0)
    return;

  if (linearizedBaseList[c_name].size() > 1)
  {
    // this means the contract is inherited from others
    // skip the first one as it's contract itself
    for (auto i_ptr = linearizedBaseList[c_name].begin() + 1;
         i_ptr != linearizedBaseList[c_name].end();
         i_ptr++)
    {
      std::string i_name = contractNamesMap[*i_ptr];
      if (linearizedBaseList[i_name].size() > 1)
      {
        if (merged_list.count(i_name) == 0)
        {
          merged_list.insert(i_name);
          merge_inheritance_ast(i_name, c_node, merged_list);
        }
        else
          // we have merged this contract
          continue;
      }

      const nlohmann::json &i_node =
        find_node_by_id(src_ast_json["nodes"], *i_ptr);
      assert(!i_node.empty());

      // abstract contract
      if (!i_node.contains("nodes"))
        continue;

      // *@i: incoming node
      // *@c_i: current node
      for (auto i : i_node["nodes"])
      {
        // skip duplicate
        bool is_dubplicate = false;
        for (const auto &c_i : c_node["nodes"])
        {
          if (c_i.contains("id") && c_i["id"] == i["id"])
          {
            is_dubplicate = true;
            break;
          }
        }
        if (is_dubplicate)
          continue;

        // Inheritance-shadow guard: when an interface declares a
        // function (no body) AND a base contract provides the
        // implementation (with body), both get copied into the
        // derived contract's nodes with DIFFERENT ids (interface id +
        // base id).  The function-by-name dispatcher resolution may
        // then bind to the empty stub instead of the real body —
        // surfaced empirically as `EscrowSrc.rescueFunds#<iface-id>`
        // having `END_FUNCTION` immediately, while
        // `BaseEscrow.rescueFunds#<base-id>` has the full body, and
        // the EscrowSrc dispatcher calling the empty stub.  Drop the
        // body-less interface declaration if we are about to merge a
        // same-name FunctionDefinition WITH body, so the dispatcher
        // resolves to the implementation.
        if (
          i.contains("nodeType") && i["nodeType"] == "FunctionDefinition" &&
          i.contains("body") && !i["body"].is_null() && i.contains("name"))
        {
          const std::string i_fname = i["name"].get<std::string>();
          for (auto c_it = c_node["nodes"].begin();
               c_it != c_node["nodes"].end();)
          {
            if (
              c_it->contains("nodeType") &&
              (*c_it)["nodeType"] == "FunctionDefinition" &&
              c_it->contains("name") &&
              (*c_it)["name"].get<std::string>() == i_fname &&
              (!c_it->contains("body") || (*c_it)["body"].is_null()))
            {
              c_it = c_node["nodes"].erase(c_it);
            }
            else
              ++c_it;
          }
        }

        // skip ctor
        if (i.contains("kind") && i["kind"].get<std::string>() == "constructor")
          continue;

        // Inherited-node filter: this merge path only handles member
        // declarations that carry a `name` (functions, state vars,
        // events, errors, etc.). Non-named nodes such as UsingForDirective
        // live in the base but are not merge candidates — skip them so
        // the `.name` access below cannot throw.
        if (!i.contains("name") || !i["name"].is_string())
          continue;

        // for virtual/override function
        std::string i_name = i["name"].get<std::string>() == ""
                               ? (i.contains("kind") && i["kind"].is_string()
                                    ? i["kind"].get<std::string>()
                                    : std::string())
                               : i["name"].get<std::string>();
        if (i_name.empty())
          continue;
        assert(!i_name.empty());
        if (i.contains("nodeType") && i["nodeType"] == "FunctionDefinition")
        {
          //! receive/fallback can be inherited but cannot be override.
          // to avoid the name ambiguous/conflict
          // order: current_contract -> most base -> derived
          bool is_conflict = false;

          assert(c_node.contains("nodes"));
          for (auto &c_i : c_node["nodes"])
          {
            if (
              c_i.contains("kind") &&
              c_i["kind"].get<std::string>() == "constructor")
              continue;

            if (
              c_i.contains("nodeType") &&
              c_i["nodeType"] == "FunctionDefinition")
            {
              assert(c_i.contains("name"));
              std::string c_iname = c_i["name"].get<std::string>() == ""
                                      ? c_i["kind"].get<std::string>()
                                      : c_i["name"].get<std::string>();
              assert(!c_iname.empty());

              if (i_name == c_iname)
              {
                /*
                   A
                  / \
                 B   C
                  \ /
                   D
                  for cases above, there must be an override inside D if B and C
                 both override A.
                */
                is_conflict = true;
                if (c_i.contains("id") && i.contains("id"))
                  overrideMap[c_name][i["id"].get<int>()] =
                    c_i["id"].get<int>();
                break;
              }
            }
          }
          if (is_conflict)
            continue;
        }

        // Here we have ruled out the special cases
        // so that we could merge the AST
        log_debug(
          "solidity",
          "\t@@@ Merging AST node {} to contract {}",
          i_name,
          c_name);
        // This is to distinguish it from the originals
        add_inherit_label(i, c_name);

        c_node["nodes"].push_back(i);
      }
    }
  }
}
