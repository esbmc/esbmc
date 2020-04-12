//
// Created by Rafael Sá Menezes on 06/04/20.
//

#ifndef ESBMC_SSA_CONTAINER_H
#define ESBMC_SSA_CONTAINER_H

#include <unordered_map>
#include <set>
#include <util/irep2.h>
#include <filesystem>

typedef size_t expr_hash;
namespace
{
struct expr_container
{
  std::set<expr_hash> expressions;

  explicit expr_container(const std::set<expr_hash> &other) : expressions(other)
  {
  }

  bool is_subset_of(const std::set<expr_hash> &other)
  {
    // Checks if 'expressions' contains all elements of 'other'
    return std::includes(
      other.begin(), other.end(), expressions.begin(), expressions.end());
  }
};

} // namespace

class green_storage
{
private:
  std::set<std::shared_ptr<expr_container>> storage;

public:
  green_storage() = default;
  ;
  virtual ~green_storage() = default;
  ;

  static expr_hash hash(const expr2t &item)
  {
    auto return_value = item.crc();
    return return_value;
  }

  bool get(const std::set<expr_hash> &items)
  {
    for(auto it : this->storage)
    {
      if(it->is_subset_of(items))
        return true;
    }
    return false;
  }
  void add(const std::set<expr_hash> &items)
  {
    this->storage.insert(std::make_shared<expr_container>(items));
  }

  std::string default_file_name = "database";
  void load()
  {
    std::ifstream infile;
    infile.open(default_file_name);

    if(!infile.is_open())
    {
      return;
    }

    std::string line;
    infile >> line;
    assert(line == "BEGIN");
    infile >> line;
    while(line != "END")
    {
      assert(line == "GUARD");
      std::set<expr_hash> items;
      infile >> line;
      while(line != "END_ITEM")
      {
        std::istringstream iss(line);
        size_t size;
        iss >> size;
        assert(size != 0);
        items.insert(size);
        infile >> line;
      }
      this->add(items);
      infile >> line;
    }
    infile.close();
  }

  void save()
  {
    std::ofstream outfile;
    outfile.open(default_file_name);
    outfile << "BEGIN" << std::endl;
    for(const auto &guard : storage)
    {
      outfile << "GUARD" << std::endl;

      for(const auto &item : guard->expressions)
      {
        outfile << item << std::endl;
        assert(item != 0);
      }
      outfile << "END_ITEM" << std::endl;
    }

    outfile << "END";
    outfile.close();
  }
};

#endif //ESBMC_SSA_CONTAINER_H
