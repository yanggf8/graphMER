from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable


@dataclass
class Entity:
    id: str
    type: str
    file: str


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    qualifiers: Dict[str, str]


class PythonCodeParser(ast.NodeVisitor):
    """Lightweight Python parser to extract ontology-aligned entities and relations.
    Extracts: Class, Function, Method, Variable (class attribute minimal), Module (by file path)
    Relations: defines/contains, inherits_from, calls/invokes, imports, declares, returns, raises, catches,
               annotated_with (decorators), instantiates (heuristic)
    """

    def __init__(self, file_path: str, module_name: str):
        self.file_path = file_path
        self.module_name = module_name
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Triple] = []
        self._class_stack: List[Tuple[str, Dict[str, ast.FunctionDef]]] = []  # (class_id, methods_by_name)
        self._fn_stack: List[str] = []  # current function/method id
        self._imports: List[str] = []
        self._is_test_file: bool = (
            "tests" in (file_path.replace("\\", "/")) or module_name.startswith("test_")
        )

    # Helpers
    def _ent(self, eid: str, etype: str) -> None:
        if eid not in self.entities:
            self.entities[eid] = Entity(id=eid, type=etype, file=self.file_path)

    def _tr(self, h: str, r: str, t: str) -> None:
        qual = {"language": "python", "file": self.file_path, "module": self.module_name}
        self.triples.append(Triple(head=h, relation=r, tail=t, qualifiers=qual))

    def _current_scope(self) -> str | None:
        return self._fn_stack[-1] if self._fn_stack else self.module_name

    # Visitors
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._imports.append(alias.name.split(".")[0])
            self._ent(self.module_name, "Module")
            self._ent(alias.name, "Module")
            self._tr(self.module_name, "imports", alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod:
            self._imports.append(mod.split(".")[0])
            self._ent(self.module_name, "Module")
            self._ent(mod, "Module")
            self._tr(self.module_name, "imports", mod)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self.module_name}.{node.name}"
        self._ent(self.file_path, "File")
        self._ent(class_id, "Class")
        self._tr(self.file_path, "defines", class_id)
        self._tr(self.file_path, "contains", class_id)

        # inherits_from
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                # best-effort fully qualified guess
                base_id = base_name if "." in base_name else f"{self.module_name}.{base_name}"
                self._ent(base_id, "Class")
                self._tr(class_id, "inherits_from", base_id)

        # Track methods
        methods: Dict[str, ast.FunctionDef] = {}
        self._class_stack.append((class_id, methods))
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Determine if this is a test function (used for tests edges only)
        is_test_fn = self._is_test_file or node.name.startswith("test_")
        if self._class_stack:
            class_id, methods = self._class_stack[-1]
            meth_id = f"{class_id}.{node.name}"
            self._ent(meth_id, "Method")
            self._tr(class_id, "declares", meth_id)
            self._tr(class_id, "contains", meth_id)
            # returns annotation
            if node.returns is not None:
                ret_name = self._get_attr_fullname(node.returns) or self._get_name(node.returns)
                if ret_name:
                    type_id = f"type::{ret_name}"
                    self._ent(type_id, "Type")
                    self._tr(meth_id, "returns", type_id)
            # decorators -> annotated_with
            for dec in node.decorator_list:
                dec_name = self._get_attr_fullname(dec) or self._get_name(dec)
                if dec_name:
                    ann_id = f"annotation::{dec_name}"
                    self._ent(ann_id, "Annotation")
                    self._tr(meth_id, "annotated_with", ann_id)
            self._fn_stack.append(meth_id)
            self.generic_visit(node)
            self._fn_stack.pop()
        else:
            fun_id = f"{self.module_name}.{node.name}"
            self._ent(self.file_path, "File")
            self._ent(fun_id, "Function")
            self._tr(self.file_path, "defines", fun_id)
            self._tr(self.file_path, "contains", fun_id)
            # returns annotation
            if node.returns is not None:
                ret_name = self._get_attr_fullname(node.returns) or self._get_name(node.returns)
                if ret_name:
                    type_id = f"type::{ret_name}"
                    self._ent(type_id, "Type")
                    self._tr(fun_id, "returns", type_id)
            for dec in node.decorator_list:
                dec_name = self._get_attr_fullname(dec) or self._get_name(dec)
                if dec_name:
                    ann_id = f"annotation::{dec_name}"
                    self._ent(ann_id, "Annotation")
                    self._tr(fun_id, "annotated_with", ann_id)
            self._fn_stack.append(fun_id)
            self.generic_visit(node)
            self._fn_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        # Attempt to resolve simple call/constructor names
        callee = self._get_attr_fullname(node.func)
        scope = self._current_scope() or self.module_name
        if callee:
            last = callee.split(".")[-1]
            # Only emit call-like relations when inside a function/method scope
            if scope != self.module_name:
                # Heuristic: Capitalized callee -> instantiation
                if last[:1].isupper():
                    cls_id = callee if "." in callee else f"{self.module_name}.{callee}"
                    self._ent(cls_id, "Class")
                    self._tr(scope, "instantiates", cls_id)
                else:
                    self._ent(callee, "Function")
                    self._tr(scope, "calls", callee)
                # invokes for external API (module-prefixed and not local module)
                first = callee.split(".")[0]
                if first in self._imports and first != self.module_name:
                    api_id = f"api::{callee}"
                    self._ent(api_id, "API")
                    self._tr(scope, "invokes", api_id)
                # if current scope name suggests a test function, emit tests edge with Test head entity
                if scope and scope.split(".")[-1].startswith("test_"):
                    test_id = f"test::{scope}"
                    self._ent(test_id, "Test")
                    self._tr(test_id, "tests", callee)
        self.generic_visit(node)

    # Utilities
    def _get_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._get_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _get_attr_fullname(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None

    def visit_Return(self, node: ast.Return) -> None:
        scope = self._current_scope()
        if scope is not None and node.value is not None:
            # attempt to resolve return type tokens; fallback to generic Type
            name = self._get_attr_fullname(node.value) or self._get_name(node.value)
            type_id = f"type::{name}" if name else "type::<value>"
            self._ent(type_id, "Type")
            self._tr(scope, "returns", type_id)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        scope = self._current_scope()
        if scope is not None and node.exc is not None:
            name = self._get_attr_fullname(node.exc) or self._get_name(node.exc)
            if name:
                exc_id = name if "." in name else f"{self.module_name}.{name}"
                self._ent(exc_id, "ExceptionType")
                self._tr(scope, "raises", exc_id)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        scope = self._current_scope()
        for h in node.handlers:
            if h.type is not None and scope is not None:
                name = self._get_attr_fullname(h.type) or self._get_name(h.type)
                if name:
                    exc_id = name if "." in name else f"{self.module_name}.{name}"
                    self._ent(exc_id, "ExceptionType")
                    self._tr(scope, "catches", exc_id)
        self.generic_visit(node)

    def parse(self, code: str) -> Tuple[Dict[str, Entity], List[Triple]]:
        tree = ast.parse(code)
        self.visit(tree)
        return self.entities, self.triples


def extract_from_file(path: str, module_name: str) -> Tuple[Dict[str, Entity], List[Triple]]:
    code = Path(path).read_text(encoding="utf-8")
    parser = PythonCodeParser(file_path=path, module_name=module_name)
    return parser.parse(code)
