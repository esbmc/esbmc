#pragma once

#include <util/message.h>
#include <string>

template <class Json>
class python_annotation
{
public:
  python_annotation(Json &ast) : ast_(ast)
  {
  }

  void add_type_annotation()
  {
    // Add type annotation in global scope variables
    add_annotation(ast_);

    // Add type annotation in function bodies
    for (Json &element : ast_["body"])
    {
      if (element["_type"] == "FunctionDef")
      {
        add_annotation(element);
        update_end_col_offset(element);
      }
    }
  }

  // Add annotation in a specific function
  void add_type_annotation(const std::string &func_name)
  {
    for (Json &elem : ast_["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        add_annotation(elem);
        update_end_col_offset(elem);
        return;
      }
    }
  }

private:
  void update_end_col_offset(Json &ast)
  {
    int max_col_offset = ast["end_col_offset"];
    for (auto &elem : ast["body"])
    {
      if (elem["end_col_offset"] > max_col_offset)
        max_col_offset = elem["end_col_offset"];
    }
    ast["end_col_offset"] = max_col_offset;
  }

  void add_annotation(Json &body)
  {
    for (auto &element : body["body"])
    {
      if (element["_type"] == "Assign" && element["type_comment"].is_null())
      {
        std::string type;
        // Get type from rhs constant
        if (element["value"]["_type"] == "Constant")
        {
          // Get type from rhs constant
          auto rhs = element["value"]["value"];
          type = get_type_from_element(rhs);
        }
        // Get type from rhs variable
        else if (element["value"]["_type"] == "Name")
        {
          // Find rhs variable declaration in the current function
          auto rhs_node = find_node(element["value"]["id"], body["body"]);

          // Find rhs variable in the function args
          if (rhs_node.empty() && body.contains("args"))
          {
            rhs_node = find_node(element["value"]["id"], body["args"]["args"]);
          }

          // Find rhs variable node in the global scope
          if (rhs_node.empty())
          {
            rhs_node = find_node(element["value"]["id"], ast_["body"]);
          }

          if (rhs_node.empty())
          {
            log_error(
              "Variable {} not found.",
              element["value"]["id"].template get<std::string>().c_str());
            abort();
          }
          type = rhs_node["annotation"]["id"];
        }
        // Get type from rhs binary expression
        else if (element["value"]["_type"] == "BinOp")
        {
          if (element["value"]["op"]["_type"] == "FloorDiv")
            type = "int";
        }
        else
          continue;

        // Update type field
        element["_type"] = "AnnAssign";

        // lhs
        auto target = element["targets"][0];
        int col_offset = target["col_offset"].template get<int>() +
                         target["id"].template get<std::string>().size() + 1;

        // Add annotation field
        element["annotation"] = {
          {"_type", target["_type"]},
          {"col_offset", col_offset},
          {"ctx", {{"_type", "Load"}}},
          {"end_col_offset", col_offset + type.size()},
          {"end_lineno", target["end_lineno"]},
          {"id", type},
          {"lineno", target["lineno"]}};

        element["end_col_offset"] =
          element["end_col_offset"].template get<int>() + type.size() + 1;
        element["end_lineno"] = element["lineno"];
        element["simple"] = 1;

        // Convert "targets" array to "target" object
        element["target"] = target;
        element.erase("targets");

        // Remove "type_comment" field
        element.erase("type_comment");

        // Update value fields
        element["value"]["col_offset"] =
          element["value"]["col_offset"].template get<int>() + type.size() + 1;
        element["value"]["end_col_offset"] =
          element["value"]["end_col_offset"].template get<int>() + type.size() +
          1;
      }
    }
  }

  std::string get_type_from_element(const Json &elem) const
  {
    if (elem.is_number_integer() || elem.is_number_unsigned())
      return std::string("int");
    else if (elem.is_boolean())
      return std::string("bool");
    else if (elem.is_number_float())
      return std::string("float");

    return std::string();
  }

  const Json find_node(const std::string &node_name, const Json &body)
  {
    for (const Json &elem : body)
    {
      if (
        elem.contains("_type") &&
        ((elem["_type"] == "AnnAssign" && elem.contains("target") &&
          elem["target"].contains("id") &&
          elem["target"]["id"].template get<std::string>() == node_name) ||
         (elem["_type"] == "arg" && elem["arg"] == node_name)))
      {
        return elem;
      }
    }
    return Json();
  }

  Json &ast_;
};
