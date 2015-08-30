template <typename T>
class Omega
{
public:
    void foo() {}

};

struct IA
{
    typedef Omega<int> OmegaInt;
    typedef Omega<char> OmegaChar;
};

int main()
{
    return 0;
}

