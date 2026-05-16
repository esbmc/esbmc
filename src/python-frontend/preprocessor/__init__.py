"""Preprocessor package facade exposing the composed Preprocessor class."""

from ast import NodeTransformer

from preprocessor.assignment_core_mixin import AssignmentCoreMixin
from preprocessor.assignment_visitors_mixin import AssignmentVisitorsMixin
from preprocessor.ast_utils_mixin import AstUtilsMixin
from preprocessor.class_context_mixin import ClassContextMixin
from preprocessor.core_visitors_mixin import CoreVisitorsMixin
from preprocessor.dataclass_mixin import DataclassMixin
from preprocessor.expression_rewrite_mixin import ExpressionRewriteMixin
from preprocessor.function_sentinel_mixin import FunctionSentinelMixin
from preprocessor.generator_mixin import GeneratorMixin
from preprocessor.helper_functions_mixin import HelperFunctionsMixin
from preprocessor.import_typing_mixin import ImportTypingMixin
from preprocessor.loop_mixin import LoopMixin
from preprocessor.module_lifecycle_mixin import ModuleLifecycleMixin
from preprocessor.module_rewrite_mixin import ModuleRewriteMixin
from preprocessor.preprocessor_state_mixin import PreprocessorStateMixin
from preprocessor.type_inference_mixin import TypeInferenceMixin

__all__ = ["Preprocessor"]


class Preprocessor(
        DataclassMixin,
        GeneratorMixin,
        HelperFunctionsMixin,
        AssignmentCoreMixin,
        AssignmentVisitorsMixin,
        CoreVisitorsMixin,
        FunctionSentinelMixin,
        ExpressionRewriteMixin,
        TypeInferenceMixin,
        ClassContextMixin,
        LoopMixin,
        ModuleRewriteMixin,
        ModuleLifecycleMixin,
        PreprocessorStateMixin,
        AstUtilsMixin,
        ImportTypingMixin,
        NodeTransformer,
):
    """Facade class composing all preprocessor mixins."""
    _YieldToAppend = GeneratorMixin._YieldToAppend
    _YieldReplacer = GeneratorMixin._YieldReplacer

    def __init__(self, module_name: str):
        super().__init__()
        self._init_preprocessor_state(module_name)
