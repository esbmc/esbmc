#pragma once

#include <nlohmann/json.hpp>

template <class JsonType>
class python_annotation
{
public:
  void add_type_annotation(nlohmann::json &ast)
  {
//    printf("\ninput:\n");
//    printf("%s\n", ast.dump(4).c_str());

    for(auto &element : ast)
    {
      // Getting type from constant on RHS
      if(
        element["_type"] == "Assign" && element["type_comment"].is_null() &&
        element["value"]["_type"] == "Constant")
      {
        //        printf("input:\n");
        //        printf("%s\n", element.dump(4).c_str());
        //        getchar();

        // Update type field
        element["_type"] = "AnnAssign";

        // Get type from rhs value
        auto rhs = element["value"]["value"];
        std::string type = get_type_from_element(rhs);

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

        // Convert targets from
        element["target"] = target;
        element.erase("targets");

        // Update value fields
        element["value"]["col_offset"] =
          element["value"]["col_offset"].get<int>() + type.size() + 1;
        element["value"]["end_col_offset"] =
          element["value"]["end_col_offset"].get<int>() + type.size() + 1;
      }
    }
//    printf("\noutput:\n");
//    printf("%s\n", ast.dump(4).c_str());
    //  exit(1);
  }

private:
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
};
