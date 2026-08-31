from __future__ import annotations
from typing import List

class Type:
    def join(self, other: 'Type') -> 'Type':
        raise NotImplementedError
    def widen(self, other: 'Type') -> 'Type':
        return self.join(other)
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return Bottom()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        # complement of narrow: remove members matching cls_name (used for false-branch)
        return self
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return False
    def to_ann_name(self) -> str:
        # canonical textual name used for AST annotations
        return 'unknown'
    def __repr__(self):
        return self.__class__.__name__

class ClassInfo:
    def __init__(
            self,
            name:str,
            bases=None,
            attributes=None,
    ):
        self.name = name
        self.bases = bases if bases is not None else []
        self.attributes = (
            attributes if attributes is not None else {}
        )
class TopType(Type):
    def join(self, other:'Type') -> 'Type':
        return self

    def widen(self, other:'Type') -> 'Type':
        return self

    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self

    def remove_isinstance(self, cls_name: str) -> 'Type':
        return self

    def is_subtype_of_name(self, cls_name: str) -> bool:
        return True

    def to_ann_name(self) -> str:
        return 'object'                    

class Unknown(Type):
    def join(self, other: 'Type') -> 'Type':
        return other
    def widen(self, other: 'Type') -> 'Type':
        return self.join(other)
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return self
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return False
    def to_ann_name(self) -> str:
        return 'unknown'

class Bottom(Type):

    def join(self, other: 'Type') -> 'Type':
        return other

    def widen(self, other: 'Type') -> 'Type':
        return other

    def narrow_with_isinstance(self, cls_name: str) -> Type:
        return self 

    def is_subtype_of_name(self, cls_name: str) -> bool:
        return True

    def remove_isinstance(self, cls_name: str) -> 'Type':
        return self

    def to_ann_name(self) -> str:
        return 'Never'

    def __repr__(self):
        return "Bottom"
    

class NoneType(Type):
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, NoneType):
            return self
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('none','nonetype','type(None)')
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self if self.is_subtype_of_name(cls_name) else Bottom()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        # remove None if cls_name refers to None
        return Bottom() if self.is_subtype_of_name(cls_name) else self
    def to_ann_name(self) -> str:
        return 'None'

class AnyType(Type):

    def join(self, other: 'Type') -> 'Type':
        return self

    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in {
            "any", "object"
        }

    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self

    def remove_isinstance(self, cls_name: str) -> 'Type':
        return self

    def to_ann_name(self) -> str:
        return 'Any'    

class BoolType(Type):
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, BoolType):
            return self
        if isinstance(other, IntType):
            return other
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('bool','int','object')
    def to_ann_name(self) -> str:
        return 'bool'
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self if self.is_subtype_of_name(cls_name) else Unknown()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Unknown() if self.is_subtype_of_name(cls_name) else self

class IntType(Type):
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, IntType) or isinstance(other, BoolType):
            return self
        if isinstance(other, FloatType):
            return other
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('int','object')
    def to_ann_name(self) -> str:
        return 'int'
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self if self.is_subtype_of_name(cls_name) else Bottom()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Bottom() if self.is_subtype_of_name(cls_name) else self

class FloatType(Type):
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, FloatType):
            return self
        if isinstance(other, IntType) or isinstance(other, BoolType):
            return self
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('float','object')
    def to_ann_name(self) -> str:
        return 'float'
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self if self.is_subtype_of_name(cls_name) else Unknown()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Unknown() if self.is_subtype_of_name(cls_name) else self

class StrType(Type):
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, StrType):
            return self
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('str','object')
    def to_ann_name(self) -> str:
        return 'str'
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self if self.is_subtype_of_name(cls_name) else Unknown()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Unknown() if self.is_subtype_of_name(cls_name) else self

class ListType(Type):
    def __init__(self, elem: Type):
        self.elem = elem
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, ListType):
            return ListType(self.elem.join(other.elem))
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def widen(self, other: 'Type') -> 'Type':
        if isinstance(other, ListType):
            return ListType(self.elem.widen(other.elem))
        if isinstance(other, UnionType):
            return other.widen(self)
        return other            
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        if cls_name.lower() in ('list','typing.list','builtins.list'):
            return self
        return Unknown()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Unknown() if cls_name.lower() in ('list','typing.list','builtins.list') else self
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('list','object','typing.list')
    def to_ann_name(self) -> str:
        return f'list[{self.elem.to_ann_name()}]'
    def __repr__(self):
        return f'List[{self.elem!r}]'

class DictType(Type):
    def __init__(self, key_t: Type, val_t: Type):
        self.key_t = key_t
        self.val_t = val_t
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, DictType):
            return DictType(self.key_t.join(other.key_t), self.val_t.join(other.val_t))
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def widen(self, other: 'Type') -> 'Type':
        if isinstance(other, DictType):
            return DictType(self.key_t.widen(other.key_t),
                            self.val_t.widen(other.val_t))
        if isinstance(other, UnionType):
            return other.widen(self)
        return other                        
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        if cls_name.lower() in ('dict','builtins.dict','typing.dict'):
            return self
        return Unknown()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Unknown() if cls_name.lower() in ('dict','builtins.dict','typing.dict') else self
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('dict','object','typing.dict')
    def to_ann_name(self) -> str:
        return f'dict[{self.key_t.to_ann_name()},{self.val_t.to_ann_name()}]'
    def __repr__(self) -> str:
        return f'Dict[{self.key_t!r}, {self.val_t!r}]'

class TupleType(Type):
    def __init__(self, elems: List[Type]):
        self.elems = elems
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, TupleType):
            if len(self.elems) == len(other.elems):
               return TupleType([a.widen(b) for a,b in zip(self.elems, other.elems)])

            return UnionType([self, other])
        if isinstance(other, UnionType):
            return other.widen(self)
        return other
    def widen(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        
        if isinstance(other, TupleType):
            if len(self.elems) == len(other.elems):
                return TupleType([a.widen(b) for a,b in zip(self.elems, other.elems)])
            return UnionType([self, other])
        if isinstance(other, UnionType):
                return other.widen(self)

        return other    
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        if cls_name.lower() in ('tuple','builtins.tuple','typing.tuple'):
            return self
        return Unknown()
    def remove_isinstance(self, cls_name: str) -> 'Type':
        return Unknown() if cls_name.lower() in ('tuple','builtins.tuple','typing.tuple') else self
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('tuple','object','typing.tuple')
    def to_ann_name(self) -> str:
        return 'tuple'
    def __repr__(self) -> str:
        return 'Tuple[' + ', '.join(repr(e) for e in self.elems) + ']'

class ComplexType(Type):
    def join(self, other: Type) -> Type:
        if isinstance(other, Unknown):
            return self

        if isinstance(other, ComplexType):
            return self

        return UnionType([self, other])

    def widen(self, other: Type) -> Type:
        if isinstance(other, ComplexType):
            return self

        return other

    def narrow_with_isinstance(self, cls_name: str) -> Type:
        if cls_name.lower() in {
            "complex",
            "builtins.complex",
        }:
            return self

        return Unknown()

    def remove_isinstance(self, cls_name: str) -> Type:
        if cls_name.lower() in {
            "complex",
            "builtins.complex",
        }:
            return Unknown()

        return self

    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in {
            "complex",
            "object",
            "builtins.complex",
        }

    def to_ann_name(self) -> str:
        return "complex"

    def __repr__(self) -> str:
        return "ComplexType"

class CallableType(Type):
    def __init__(self, param_types: List[Type], ret: Type):
        self.param_types = param_types
        self.ret = ret
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, CallableType):
            pt = [a.join(b) for a,b in zip(self.param_types, other.param_types)]
            return CallableType(pt, self.ret.join(other.ret))
        if isinstance(other, UnionType):
            return other.join(self)
        return UnionType([self, other])
    def widen(self, other: 'Type') -> 'Type':
        if isinstance(other, CallableType) and len(self.param_types) == len(other.param_types):
            return CallableType([a.widen(b) for a, b in zip(self.param_types, other.param_types)],
            self.ret.widen(other.ret))
        return TopType()    
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in ('callable','typing.callable','object')
    def to_ann_name(self) -> str:
        p = ','.join(p.to_ann_name() for p in self.param_types)
        return f'Callable[{p}->{self.ret.to_ann_name()}]'
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        return self if self.is_subtype_of_name(cls_name) else Unknown()

class InstanceType(Type):
    def __init__(self, class_name:str):
        self.class_name = class_name

    def join(self, other):
        if isinstance(other, InstanceType):
            if self.class_name == other.class_name:
                return self
        return UnionType([self, other])

    def __repr__(self):
        return f"Instance[{self.class_name}]"

    def to_ann_name(self) -> str:
        return self.class_name  

class SetType(Type):
    def __init__(self, elem: Type):
        self.elem = elem

    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self

        if isinstance(other, SetType):
            return SetType(self.elem.join(other.elem))

        if isinstance(other, UnionType):
            return other.join(self)

        return UnionType([self, other])

    def widen(self, other: 'Type') -> 'Type':
        if isinstance(other, SetType):
            return SetType(
                self.elem.widen(other.elem)
            )

        if isinstance(other, UnionType):
            return other.widen(self)

        return other

    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        if cls_name.lower() in {
            'set',
            'builtins.set',
            'typing.set',
        }:
            return self

        return Unknown()

    def remove_isinstance(self, cls_name: str) -> 'Type':
        if cls_name.lower() in {
            'set',
            'builtins.set',
            'typing.set',
        }:
            return Unknown()

        return self

    def is_subtype_of_name(self, cls_name: str) -> bool:
        return cls_name.lower() in {
            'set',
            'builtins.set',
            'typing.set',
            'object',
        }

    def to_ann_name(self) -> str:
        return f'set[{self.elem.to_ann_name()}]'

    def __repr__(self) -> str:
        return f'Set[{self.elem!r}]'
                      

class UnionType(Type):
    def __init__(self, members: List[Type]):
        flat = []
        for m in members:
            if isinstance(m, UnionType):
                flat.extend(m.members)
            else:
                flat.append(m)
        seen = set()
        out = []
        for m in flat:
            k = repr(m)
            if k not in seen:
                seen.add(k)
                out.append(m)
        self.members = out
    def join(self, other: 'Type') -> 'Type':
        if isinstance(other, Unknown):
            return self
        if isinstance(other, UnionType):
            return UnionType(self.members + other.members)
        # if any(isinstance(m, FloatType) for m in self.members) and isinstance(other, IntType):
        #     new_members: List[Type] = []

        #     for member in self.members:
        #         if isinstance(member, IntType):
        #             new_members.append(FloatType())
        #         else:
        #             new_members.append(member)   
        #     return UnionType(new_members)
        return UnionType(self.members + [other])
    
    def widen(self, other: 'Type') -> 'Type':
        return self.join(other)
    def narrow_with_isinstance(self, cls_name: str) -> 'Type':
        kept = []
        for m in self.members:
            if m.is_subtype_of_name(cls_name):
                kept.append(m)
            else:
                nm = m.narrow_with_isinstance(cls_name)
                if not isinstance(nm, Unknown):
                    kept.append(nm)
        if not kept:
            return Unknown()
        if len(kept) == 1:
            return kept[0]
        return UnionType(kept)
    def remove_isinstance(self, cls_name: str) -> 'Type':
        kept = []
        for m in self.members:
            if not m.is_subtype_of_name(cls_name):
                rm = m.remove_isinstance(cls_name)
                if not isinstance(rm, Unknown):
                    kept.append(rm)
        if not kept:
            return Unknown()
        if len(kept) == 1:
            return kept[0]
        return UnionType(kept)
    def is_subtype_of_name(self, cls_name: str) -> bool:
        return all(m.is_subtype_of_name(cls_name) for m in self.members)
    def to_ann_name(self) -> str:
        return 'Union[' + ','.join(m.to_ann_name() for m in self.members) + ']'
    def __repr__(self) -> str:
        return 'Union[' + ', '.join(repr(m) for m in self.members) + ']'

def mk_type_from_name(name: str) -> Type:
    n = name.lower()
    if n == 'int':
        return IntType()
    if n == 'float':
        return FloatType()
    if n == 'bool':
        return BoolType()
    if n == 'none' or n=='nonetype' or n=='type(None)':
        return NoneType()
    if n == 'str':
        return StrType()
    if n == 'unknown':
        return Unknown()
    if n == 'set':
        return SetType(Unknown())
    if n == 'tuple':
        return TupleType([])

    if n.startswith('set['):
        try:
            inner = name[name.find('[') + 1: -1]
            return SetType(mk_type_from_name(inner))
        except Exception:
            return SetType(Unknown())
    if n.startswith('list['):
        try:
            inner = name[name.find('[')+1:-1]
            return ListType(mk_type_from_name(inner))
        except Exception:
            return ListType(Unknown())
    if n.startswith('tuple['):
        try:
            inner = name[name.find('[')+1:-1]
            parts = [part.strip() for part in inner.split(',')]
            elems = [mk_type_from_name(part) for part in parts]
            return TupleType(elems)
        except Exception:
            return TupleType([])   
    if n.startswith('dict['):
        try:
            inner = name[name.find('[')+1:-1]
            k,v = inner.split(',')
            return DictType(mk_type_from_name(k.strip()), mk_type_from_name(v.strip()))
        except Exception:
            return DictType(Unknown(), Unknown())
    return Unknown()
