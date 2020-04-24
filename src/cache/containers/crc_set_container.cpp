// Rafael Sá Menezes - 04/2020

#include <cache/containers/crc_set_container.h>
#include <sstream>
#include <util/irep2.h>
#include <iostream>
#include <fstream>

bool expr_set_container::is_subset_of(const std::set<long> &other)
{
  return std::includes(
    other.begin(),
    other.end(),
    this->expression.begin(),
    this->expression.end());
}

bool ssa_set_container::check(const std::set<long> &items)
{
  for(auto it : this->expressions)
  {
    if(it->is_subset_of(items))
      return true;
  }
  return false;
}

void ssa_set_container::add(const std::set<long> &items)
{
  this->expressions.insert(std::make_shared<expr_set_container>(items));
}

ssa_container<ssa_container_type> text_file_crc_set_storage::load()
{
  std::ifstream infile;
  infile.open(filename);
  if(!infile.is_open())
  {
    return ssa_set_container();
  }
  auto result = load(infile);
  infile.close();
  return result;
}

ssa_container<ssa_container_type>
text_file_crc_set_storage::load(std::istream &infile)
{
  ssa_set_container container;

  std::string line;
  infile >> line;
  assert(line == "BEGIN");
  infile >> line;
  while(line != "END")
  {
    assert(line == "GUARD");
    crc_expr items;
    infile >> line;
    while(line != "END_ITEM")
    {
      std::istringstream iss(line);
      crc_hash size;
      iss >> size;
      assert(size != 0);
      items.insert(size);
      infile >> line;
    }
    container.add(items);
    infile >> line;
  }
  return container;
}
void text_file_crc_set_storage::store(ssa_container<ssa_container_type> &output)
{
  std::ofstream outfile;
  outfile.open(filename);
  outfile << "BEGIN" << std::endl;
  for(const auto &guard : output.get())
  {
    outfile << "GUARD" << std::endl;

    for(const auto &item : guard->get())
    {
      outfile << item << std::endl;
      assert(item != 0);
    }
    outfile << "END_ITEM" << std::endl;
  }

  outfile << "END";
  outfile.close();
}