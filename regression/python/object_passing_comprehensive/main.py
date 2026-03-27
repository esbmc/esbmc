# Comprehensive test for passing objects with forward-referenced types
# Tests pointer-to-struct semantics for Python objects passed as parameters

# Test 1: Basic object passing with single attribute
class Point:
    def __init__(self, x: int) -> None:
        self.x: int = x
    
    def create_vector(self) -> 'Vector':
        return Vector(self)

class Vector:
    def __init__(self, p: Point) -> None:
        self.x: int = p.x  # Access attribute from parameter object

# Test basic object passing
p1 = Point(10)
v1 = p1.create_vector()
assert v1.x == 10


# Test 2: Object passing with multiple attributes
class Rectangle:
    def __init__(self, width: int, height: int) -> None:
        self.width: int = width
        self.height: int = height
    
    def to_box(self) -> 'Box':
        return Box(self)

class Box:
    def __init__(self, rect: Rectangle) -> None:
        self.width: int = rect.width
        self.height: int = rect.height
        self.area: int = rect.width * rect.height

rect = Rectangle(5, 8)
box = rect.to_box()
assert box.width == 5
assert box.height == 8
assert box.area == 40


class DataA:
    def __init__(self, value: int) -> None:
        self.value: int = value
    
    def to_b(self) -> 'DataB':
        return DataB(self)

class DataB:
    def __init__(self, a: DataA) -> None:
        self.value: int = a.value
        self.doubled: int = a.value * 2
    
    def to_c(self) -> 'DataC':
        return DataC(self)

class DataC:
    def __init__(self, b: DataB) -> None:
        self.original: int = b.value
        self.doubled: int = b.doubled
        self.tripled: int = b.value * 3

data_a = DataA(7)
data_b = data_a.to_b()
assert data_b.value == 7
assert data_b.doubled == 14

data_c = data_b.to_c()
assert data_c.original == 7
assert data_c.doubled == 14
assert data_c.tripled == 21


# Test 4: Multiple parameter objects
class Source1:
    def __init__(self, x: int) -> None:
        self.x: int = x

class Source2:
    def __init__(self, y: int) -> None:
        self.y: int = y

class Merger:
    def __init__(self, s1: Source1, s2: Source2) -> None:
        self.x: int = s1.x
        self.y: int = s2.y
        self.sum: int = s1.x + s2.y

src1 = Source1(10)
src2 = Source2(20)
merged = Merger(src1, src2)
assert merged.x == 10
assert merged.y == 20
assert merged.sum == 30


# Test 5: Object with computed attributes
class Temperature:
    def __init__(self, celsius: int) -> None:
        self.celsius: int = celsius
    
    def to_converter(self) -> 'TempConverter':
        return TempConverter(self)

class TempConverter:
    def __init__(self, temp: Temperature) -> None:
        self.celsius: int = temp.celsius
        self.fahrenheit: int = (temp.celsius * 9) // 5 + 32  # Integer division for simplicity

temp = Temperature(0)
converter = temp.to_converter()
assert converter.celsius == 0
assert converter.fahrenheit == 32

temp2 = Temperature(100)
converter2 = temp2.to_converter()
assert converter2.celsius == 100
assert converter2.fahrenheit == 212


# Test 6: Nested object creation with method chains
class Node:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def create_wrapper(self) -> 'Wrapper':
        return Wrapper(self)

class Wrapper:
    def __init__(self, node: Node) -> None:
        self.val: int = node.val
        self.wrapped_val: int = node.val + 100
    
    def create_container(self) -> 'Container':
        return Container(self)

class Container:
    def __init__(self, wrapper: Wrapper) -> None:
        self.original: int = wrapper.val
        self.wrapped: int = wrapper.wrapped_val
        self.total: int = wrapper.val + wrapper.wrapped_val

node = Node(5)
wrapper = node.create_wrapper()
assert wrapper.val == 5
assert wrapper.wrapped_val == 105

container = wrapper.create_container()
assert container.original == 5
assert container.wrapped == 105
assert container.total == 110


# Test 7: Object passing with parameter validation
class Positive:
    def __init__(self, value: int) -> None:
        self.value: int = value
    
    def to_doubled(self) -> 'Doubled':
        return Doubled(self)

class Doubled:
    def __init__(self, pos: Positive) -> None:
        self.original: int = pos.value
        self.result: int = pos.value * 2

pos = Positive(15)
doubled = pos.to_doubled()
assert doubled.original == 15
assert doubled.result == 30


# Test 8: Complex chain with multiple transformations
class Input:
    def __init__(self, base: int) -> None:
        self.base: int = base

class Processor:
    def __init__(self, inp: Input) -> None:
        self.base: int = inp.base
        self.processed: int = inp.base * 2

class Output:
    def __init__(self, proc: Processor) -> None:
        self.original: int = proc.base
        self.intermediate: int = proc.processed
        self.final: int = proc.processed + 10

inp = Input(20)
proc = Processor(inp)
assert proc.base == 20
assert proc.processed == 40

out = Output(proc)
assert out.original == 20
assert out.intermediate == 40
assert out.final == 50


# Test 9: Boundary values - zero
class ZeroValue:
    def __init__(self) -> None:
        self.value: int = 0
    
    def to_holder(self) -> 'ZeroHolder':
        return ZeroHolder(self)

class ZeroHolder:
    def __init__(self, zv: ZeroValue) -> None:
        self.original: int = zv.value
        self.incremented: int = zv.value + 1

zero = ZeroValue()
zero_holder = zero.to_holder()
assert zero_holder.original == 0
assert zero_holder.incremented == 1


# Test 10: Boundary values - negative numbers
class NegativeValue:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def to_absolute(self) -> 'AbsoluteValue':
        return AbsoluteValue(self)

class AbsoluteValue:
    def __init__(self, neg: NegativeValue) -> None:
        self.original: int = neg.val
        # Simple absolute value for negative integers
        self.absolute: int = -neg.val if neg.val < 0 else neg.val

neg = NegativeValue(-15)
abs_val = neg.to_absolute()
assert abs_val.original == -15
assert abs_val.absolute == 15


class LevelA:
    def __init__(self, base: int) -> None:
        self.value: int = base
    
    def to_b(self) -> 'LevelB':
        return LevelB(self)

class LevelB:
    def __init__(self, a: LevelA) -> None:
        self.value: int = a.value + 1
    
    def to_c(self) -> 'LevelC':
        return LevelC(self)

class LevelC:
    def __init__(self, b: LevelB) -> None:
        self.value: int = b.value + 1
    
    def to_d(self) -> 'LevelD':
        return LevelD(self)

class LevelD:
    def __init__(self, c: LevelC) -> None:
        self.value: int = c.value + 1
    
    def to_e(self) -> 'LevelE':
        return LevelE(self)

class LevelE:
    def __init__(self, d: LevelD) -> None:
        self.value: int = d.value + 1
        self.total_increments: int = 4  # Incremented 4 times from A to E

level_a = LevelA(10)
level_b = level_a.to_b()
assert level_b.value == 11

level_c = level_b.to_c()
assert level_c.value == 12

level_d = level_c.to_d()
assert level_d.value == 13

level_e = level_d.to_e()
assert level_e.value == 14
assert level_e.total_increments == 4


# Test 12: Same-type object passing (linked list node)
class ListNode:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def create_next(self, next_val: int) -> 'ListNode':
        return ListNode(next_val)

node1 = ListNode(1)
node2 = node1.create_next(2)
node3 = node2.create_next(3)

assert node1.val == 1
assert node2.val == 2
assert node3.val == 3


# Test 13: Object independence after passing
class Original:
    def __init__(self, x: int) -> None:
        self.x: int = x
    
    def make_copy(self) -> 'Copy':
        return Copy(self)

class Copy:
    def __init__(self, orig: Original) -> None:
        self.x: int = orig.x
        self.source_value: int = orig.x

orig = Original(100)
copy = orig.make_copy()
assert copy.x == 100
assert copy.source_value == 100
# Verify original is unchanged
assert orig.x == 100


# Test 14: Multiple attributes with different types of operations
class ComplexSource:
    def __init__(self, a: int, b: int, c: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c
    
    def transform(self) -> 'ComplexTarget':
        return ComplexTarget(self)

class ComplexTarget:
    def __init__(self, src: ComplexSource) -> None:
        self.sum: int = src.a + src.b + src.c
        self.product: int = src.a * src.b
        self.difference: int = src.a - src.b
        self.max_val: int = src.a if src.a > src.b else src.b

complex_src = ComplexSource(10, 5, 3)
complex_target = complex_src.transform()
assert complex_target.sum == 18
assert complex_target.product == 50
assert complex_target.difference == 5
assert complex_target.max_val == 10


# Test 15: Diamond-shaped dependency
class Base:
    def __init__(self, val: int) -> None:
        self.val: int = val

class Left:
    def __init__(self, base: Base) -> None:
        self.left_val: int = base.val * 2

class Right:
    def __init__(self, base: Base) -> None:
        self.right_val: int = base.val * 3

class Diamond:
    def __init__(self, left: Left, right: Right) -> None:
        self.left: int = left.left_val
        self.right: int = right.right_val
        self.combined: int = left.left_val + right.right_val

base = Base(5)
left = Left(base)
right = Right(base)
diamond = Diamond(left, right)

assert diamond.left == 10
assert diamond.right == 15
assert diamond.combined == 25


# Test 16: Object with default values and optional initialization
class DefaultValues:
    def __init__(self, provided: int) -> None:
        self.provided: int = provided
        self.default_zero: int = 0
        self.default_one: int = 1
    
    def create_derived(self) -> 'DerivedFromDefaults':
        return DerivedFromDefaults(self)

class DerivedFromDefaults:
    def __init__(self, defaults: DefaultValues) -> None:
        self.from_provided: int = defaults.provided
        self.from_zero: int = defaults.default_zero
        self.from_one: int = defaults.default_one
        self.sum: int = defaults.provided + defaults.default_zero + defaults.default_one

defaults = DefaultValues(42)
derived = defaults.create_derived()
assert derived.from_provided == 42
assert derived.from_zero == 0
assert derived.from_one == 1
assert derived.sum == 43


# Test 17: Large value handling
class LargeValue:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def scale(self) -> 'ScaledValue':
        return ScaledValue(self)

class ScaledValue:
    def __init__(self, large: LargeValue) -> None:
        self.original: int = large.val
        self.half: int = large.val // 2

large = LargeValue(1000)
scaled = large.scale()
assert scaled.original == 1000
assert scaled.half == 500


# Test 18: Consecutive transformations with state accumulation
class Counter:
    def __init__(self, start: int, steps: int) -> None:
        self.count: int = start
        self.steps: int = steps
    
    def increment(self) -> 'CounterIncremented':
        return CounterIncremented(self)

class CounterIncremented:
    def __init__(self, prev: Counter) -> None:
        self.count: int = prev.count + 1
        self.steps: int = prev.steps + 1
    
    def increment(self) -> 'CounterIncremented':
        return CounterIncremented(self)

c0 = Counter(0, 0)
assert c0.count == 0
assert c0.steps == 0

c1 = c0.increment()
assert c1.count == 1
assert c1.steps == 1

c2 = c1.increment()
assert c2.count == 2
assert c2.steps == 2

c3 = c2.increment()
assert c3.count == 3
assert c3.steps == 3

