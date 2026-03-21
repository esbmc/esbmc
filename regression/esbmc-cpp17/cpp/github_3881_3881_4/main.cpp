#include <iostream>
#include <memory>

class Coord
{
  public:
  
    virtual ~Coord() = default;
    virtual std::unique_ptr<Coord> clone() const = 0;
    virtual void assign( const Coord& other ) = 0;

    Coord& operator=( const Coord& other )
    {
        assign( other );
        return *this;
    }
  
    virtual void print( std::ostream& os ) const = 0;
  
    friend std::ostream& operator<<(std::ostream& os, const Coord& coord)
    {
      coord.print( os );
      return os;
    }
 
    virtual void inc() = 0;
};

class IntCoord : public Coord
{
  private:
    int x, y;
  
  public:
    IntCoord( int xnum, int ynum ):
      x( xnum ),
      y( ynum )
      {}

    std::unique_ptr<Coord> clone() const override {
        return std::make_unique<IntCoord>(*this);
    }

    void assign( const Coord& other ) override
    {
      // throws std::bad_cast if types differ
      const auto& o = dynamic_cast<const IntCoord&>( other );
      
      x = o.x;
      y = o.y;
    }

    void print( std::ostream& os ) const override
    {
      os << "(" << x <<","<< y <<")";
    }
      
    void inc() override
      { ++x; ++y; }

};

int main()
{
  std::unique_ptr<Coord> coord0 = std::make_unique<IntCoord>(17,17);
  std::unique_ptr<Coord> coord1 = std::make_unique<IntCoord>(13,13);
  coord0->inc();
  *coord1 = *coord0;
  std::cout << "coord0 = " << *coord0 << std::endl;
  std::cout << "coord1 = " << *coord1 << std::endl;
  __ESBMC_assert( *coord0 == *coord1, "==" );
  return 0;
}
