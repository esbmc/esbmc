#pragma once

#include <nlohmann/json.hpp>
#include <util/message.h>
#include <string>

template <class JsonType>
class python_annotation
{
public:
  python_annotation(nlohmann::json &ast) : ast_(ast)
  {
  }

  void add_type_annotation()
  {
    // Add type annotation in global scope variables
    add_annotation(ast_);

    // Add type annotation in function bodies
    for(auto &element : ast_["body"])
    {
      if(element["_type"] == "FunctionDef")
      {
        add_annotation(element);
        update_end_col_offset(element);
      }
    }
  }

private:
  void update_end_col_offset(nlohmann::json &ast)
  {
    int max_col_offset = ast["end_col_offset"];
    for(auto &elem : ast["body"])
    {
      if(elem["end_col_offset"] > max_col_offset)
        max_col_offset = elem["end_col_offset"];
    }
    ast["end_col_offset"] = max_col_offset;
  }

  void add_annotation(nlohmann::json &body)
  {
    for(auto &element : body["body"])
    {
      if(element["_type"] == "Assign" && element["type_comment"].is_null())
      {
        std::string type;
        // Get type from rhs constant
        if(element["value"]["_type"] == "Constant")
        {
          // Get type from rhs constant
          auto rhs = element["value"]["value"];
          type = get_type_from_element(rhs);
        }
        // Get type from rhs variable
        else if(element["value"]["_type"] == "Name")
        {
          // Try find rhs variable node in the global scope
          auto rhs_node = find_node(element["value"]["id"], ast_["body"]);

          if(rhs_node.empty())
          {
            log_error(
              "Variable {} not found.",
              element["value"]["id"].get<std::string>().c_str());
            abort();
          }
          type = rhs_node["annotation"]["id"];
        }
        else
          continue;

        // Update type field
        element["_type"] = "AnnAssign";

        // lhs
        auto target = element["targets"][0];
        int col_offset = target["col_offset"].get<int>() +
                         strlen(target["id"].get<std::string>().c_str()) + 1;

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
          element["end_col_offset"].get<int>() + type.size() + 1;
        element["end_lineno"] = element["lineno"];
        element["simple"] = 1;

        // Convert "targets" array to "target" object
        element["target"] = target;
        element.erase("targets");

        // Remove "type_comment" field
        element.erase("type_comment");

        // Update value fields
        element["value"]["col_offset"] =
          element["value"]["col_offset"].get<int>() + type.size() + 1;
        element["value"]["end_col_offset"] =
          element["value"]["end_col_offset"].get<int>() + type.size() + 1;

        type = "";
      }
    }
  }

  std::string get_type_from_element(const JsonType &elem) const
  {
    if(elem.is_number_integer() || elem.is_number_unsigned())
      return std::string("int");
    else if(elem.is_boolean())
      return std::string("bool");
    else if(elem.is_number_float())
      return std::string("float");

    return std::string();
  }

  const JsonType find_node(const std::string &node_name, const JsonType &body)
  {
    for(const auto &elem : body)
    {
      if(
        elem["_type"] == "AnnAssign" &&
        elem["target"]["id"].template get<std::string>() == node_name)
        return elem;
    }
    return nlohmann::json();
  }

  nlohmann::json &ast_;
};
