from __future__ import annotations
import javalang
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


class JavaCodeParser:
    """AST-based Java parser to extract ontology-aligned entities and relations."""

    def __init__(self, file_path: str, package_name: str = ''):
        self.file_path = file_path
        self.package_name = package_name
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Triple] = []
        self._class_stack: List[str] = []
        self._method_stack: List[str] = []
        self._imports: List[str] = []

    def _ent(self, eid: str, etype: str) -> None:
        if eid not in self.entities:
            self.entities[eid] = Entity(id=eid, type=etype, file=self.file_path)

    def _tr(self, h: str, r: str, t: str) -> None:
        qual = {"language": "java", "file": self.file_path, "package": self.package_name}
        self.triples.append(Triple(head=h, relation=r, tail=t, qualifiers=qual))

    def _current_scope(self) -> str | None:
        if self._method_stack:
            return self._method_stack[-1]
        if self._class_stack:
            return self._class_stack[-1]
        return self.package_name

    def parse(self, code: str) -> Tuple[Dict[str, Entity], List[Triple]]:
        try:
            tree = javalang.parse.parse(code)
        except javalang.tokenizer.LexerError:
            # Skip files with syntax errors
            return {}, []

        # Pass 1: Imports and package
        if tree.package:
            self.package_name = tree.package.name
        for imp in tree.imports:
            self._imports.append(imp.path)
            self._ent(self.package_name or self.file_path, "Package")
            self._ent(imp.path, "Package")
            self._tr(self.package_name or self.file_path, "imports", imp.path)

        # Pass 2: Types and methods
        for _, node in tree.filter(javalang.tree.TypeDeclaration):
            self.visit_TypeDeclaration(node)
        
        # Pass 3: Invocations
        for _, node in tree.filter(javalang.tree.MethodInvocation):
            self.visit_MethodInvocation(node)

        return self.entities, self.triples

    def visit_TypeDeclaration(self, node: javalang.tree.TypeDeclaration):
        if isinstance(node, javalang.tree.ClassDeclaration):
            class_id = f"{self.package_name}.{node.name}"
            self._ent(self.file_path, "File")
            self._ent(class_id, "Class")
            self._tr(self.file_path, "defines", class_id)
            self._tr(self.file_path, "contains", class_id)

            if node.extends:
                base_name = node.extends.name
                self._ent(base_name, "Class")
                self._tr(class_id, "inherits_from", base_name)

            if node.implements:
                for impl in node.implements:
                    self._ent(impl.name, "Interface")
                    self._tr(class_id, "implements", impl.name)
            
            self._class_stack.append(class_id)
            for method in node.methods:
                self.visit_MethodDeclaration(method, class_id)
            self._class_stack.pop()

    def visit_MethodDeclaration(self, node: javalang.tree.MethodDeclaration, class_id: str):
        meth_id = f"{class_id}.{node.name}"
        self._ent(meth_id, "Method")
        self._tr(class_id, "declares", meth_id)
        self._tr(class_id, "contains", meth_id)

        if node.return_type:
            ret_name = node.return_type.name
            type_id = f"type::{ret_name}"
            self._ent(type_id, "Type")
            self._tr(meth_id, "returns", type_id)
        
        for annotation in node.annotations:
            ann_id = f"annotation::{annotation.name}"
            self._ent(ann_id, "Annotation")
            self._tr(meth_id, "annotated_with", ann_id)

    def visit_MethodInvocation(self, node: javalang.tree.MethodInvocation):
        scope = self._current_scope()
        if not scope:
            return

        callee_name = node.member
        qualifier = node.qualifier or ''
        
        # Heuristic for full callee path
        full_callee = f"{qualifier}.{callee_name}" if qualifier else callee_name
        
        self._ent(full_callee, "Method")
        self._tr(scope, "calls", full_callee)

        # Check if it's an external API call
        for imp in self._imports:
            if qualifier and imp.endswith(qualifier):
                api_id = f"api::{imp}.{callee_name}"
                self._ent(api_id, "API")
                self._tr(scope, "invokes", api_id)
                break


def extract_from_file(path: str) -> Tuple[Dict[str, Entity], List[Triple]]:
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    parser = JavaCodeParser(file_path=path)
    return parser.parse(code)