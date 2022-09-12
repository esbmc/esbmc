#ifndef _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_
#define _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_

#include <memory>
#include <string>

class crypto_hash_private;

class crypto_hash
{
public:
  std::shared_ptr<crypto_hash_private> p_crypto;
  unsigned int hash[5];

  bool operator<(const crypto_hash &h2) const;

  std::array<unsigned int, 5> to_array() const
  {
    // TODO: CPP+20 will have a to_array function
    std::array<unsigned int, 5> result;
    std::copy(hash, hash + 5, result.begin());
    return result;
  }

  size_t to_size_t() const
  {
    size_t result = hash[0];
    for(int i = 1; i < 5; i++)
      // Do we care about overlaps?
      result ^= (hash[i] << sizeof(unsigned int));
    return result;
  }

  std::string to_string() const;

  crypto_hash();
  void ingest(void const *data, unsigned int size);
  void fin();
};

#endif /* _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_ */
