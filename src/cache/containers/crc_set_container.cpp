// Rafael Sá Menezes - 04/2020

#include <cache/containers/crc_set_container.h>
#include <cache/containers/bloom_filter.h>
#include <cache/containers/lru_cache.h>
#include <sstream>
#include <fstream>
#include <array>

namespace
{
using guards_crc = crc_expr;

using crc_hash_function = bloom_filter_function<guards_crc>;
using hash_set = std::set<crc_hash_function>;
template <size_t N, size_t K>
using crc_bloom_filter = bloom_filter<guards_crc, N, K>;
using crc_lru = lru_cache<crc_expr>;

// Bloom Filter

/*
 * These variables were found using bloom filter formulas to determine the
 * theoretical best values. From there empirical evaluation was used to find
 * the best values.
 */
const size_t bloom_filter_size = 100000;
const size_t number_of_hashes = 3;

crc_hash_function f1([](const guards_crc &t) {
  size_t value = 0;
  for(const auto &i : t)
  {
    value += i;
  }
  return (size_t)0;
});

const crc_hash_function f2([](const guards_crc &t) {
  size_t value = 1;
  for(const auto &i : t)
  {
    value *= i;
  }
  return (size_t)value;
});
const crc_hash_function f3([](const guards_crc &t) {
  size_t value = 1;
  for(const auto &i : t)
  {
    value *= i;
  }
  return (size_t)pow(2, value);
});

std::array<crc_hash_function, number_of_hashes> hash_functions = {f1, f2, f3};

crc_bloom_filter<bloom_filter_size, number_of_hashes> filter(hash_functions);

// LRU Cache

const size_t lru_cache_size = 100;
crc_lru cache(lru_cache_size);

} // namespace

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
  if(!filter.test_element(items))
    return false;
  if(cache.exists(items))
    return true;
  for(auto it : this->expressions)
  {
    if(it->is_subset_of(items))
    {
      cache.insert(items);
      return true;
    }
  }
  return false;
}

void ssa_set_container::add(const std::set<long> &items)
{
  filter.insert_element(items);
  cache.insert(items);
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