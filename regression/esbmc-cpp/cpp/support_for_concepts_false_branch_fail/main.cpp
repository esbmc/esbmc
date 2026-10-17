template <typename T>
concept HasValue = requires(T t) {
    t.value;
};

struct S {};

int main()
{
  __ESBMC_assert(HasValue<S>, "");
}
