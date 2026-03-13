class Color
{
public:
  enum class Enum: int {RED, GREEN, YELLOW };

  static const int CARD0;
  static const int CARD1;
  static const int CARD2;

private:
  Enum enval;

  explicit constexpr Color(Enum val): enval(val) {}

  static constexpr int priv_to_int(Enum enval)
  {
    return static_cast<int>(enval);
  }

public:
  Color();

  static constexpr int pub_to_int(Enum enval)
  {
    return static_cast<int>(enval);
  }

  constexpr Enum to_enum() const
  {
    return enval;
  }

  static const Color LAST;
};

constexpr Color Color::LAST = Color(Color::Enum::YELLOW);
constexpr int Color::CARD0 = priv_to_int(Color::LAST.to_enum());
constexpr int Color::CARD1 = pub_to_int(Color::LAST.to_enum());
constexpr int Color::CARD2 = Color::pub_to_int(Color(Color::Enum::YELLOW).to_enum());
const int CARD3 = Color::pub_to_int(Color::LAST.to_enum());

int main()
{
  __ESBMC_assert(Color::pub_to_int(Color::LAST.to_enum()) == 2, "card-expr");
  __ESBMC_assert(Color::CARD0 == 2, "card0");
  __ESBMC_assert(Color::CARD1 == 2, "card1");
  __ESBMC_assert(Color::CARD2 == 2, "card2");
  __ESBMC_assert(CARD3 == 2, "card3");
  return 0;
}
