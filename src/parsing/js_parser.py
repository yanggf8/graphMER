from __future__ import annotations
import esprima
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

class JavaScriptCodeParser:
    """AST-based JavaScript parser for entity and relation extraction."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Triple] = []
        self._scope_stack: List[str] = []

    def _ent(self, eid: str, etype: str) -> None:
        if eid not in self.entities:
            self.entities[eid] = Entity(id=eid, type=etype, file=self.file_path)

    def _tr(self, h: str, r: str, t: str) -> None:
        qual = {"language": "javascript", "file": self.file_path}
        self.triples.append(Triple(head=h, relation=r, tail=t, qualifiers=qual))

    def _current_scope(self) -> str:
        return self._scope_stack[-1] if self._scope_stack else self.file_path

    def parse(self, code: str) -> Tuple[Dict[str, Entity], List[Triple]]:
        try:
            # Use parseModule to support import/export syntax
            tree = esprima.parseModule(code, {'loc': True})
        except esprima.Error:
            return {}, [] # Skip files with syntax errors
        
        self._traverse(tree)
        return self.entities, self.triples

    def _traverse(self, node):
        if not hasattr(node, 'type'):
            return

        # Pre-order traversal: process node, then children
        visitor_method = getattr(self, f'visit_{node.type}', None)
        if visitor_method:
            visitor_method(node)

        for key, value in node.__dict__.items():
            if key == 'type': continue
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'type'):
                        self._traverse(item)
            elif hasattr(value, 'type'):
                self._traverse(value)

    def visit_FunctionDeclaration(self, node: esprima.nodes.Node):
        if node.id and node.id.name:
            func_id = node.id.name
            scope = self._current_scope()
            self._ent(func_id, "Function")
            self._tr(scope, "defines", func_id)
            self._scope_stack.append(func_id)
            self._traverse(node.body)
            self._scope_stack.pop()

    def visit_VariableDeclarator(self, node: esprima.nodes.Node):
        if node.id.type == 'Identifier':
            var_id = node.id.name
            scope = self._current_scope()
            self._ent(var_id, "Variable")
            self._tr(scope, "declares", var_id)

            # Check for function expressions
            if node.init and node.init.type in ('FunctionExpression', 'ArrowFunctionExpression'):
                self._ent(var_id, "Function")
                self._scope_stack.append(var_id)
                self._traverse(node.init.body)
                self._scope_stack.pop()

    def visit_CallExpression(self, node: esprima.nodes.Node):
        scope = self._current_scope()
        callee_name = self._get_callee_name(node.callee)
        if callee_name:
            self._ent(callee_name, "Function") # Assume it's a function for now
            self._tr(scope, "calls", callee_name)

    def visit_ImportDeclaration(self, node: esprima.nodes.Node):
        source = node.source.value
        scope = self._current_scope()
        self._ent(source, "Module")
        self._tr(scope, "imports", source)
        for specifier in node.specifiers:
            if specifier.type == 'ImportSpecifier':
                local_name = specifier.local.name
                imported_name = specifier.imported.name
                self._ent(local_name, "Variable")
                self._tr(source, "exports", imported_name)
                self._tr(imported_name, "alias_of", local_name)

    def visit_ExportDefaultDeclaration(self, node: esprima.nodes.Node):
        scope = self._current_scope()
        if node.declaration and node.declaration.type == 'Identifier':
            exported_name = node.declaration.name
            self._tr(scope, "exports", exported_name)

    def _get_callee_name(self, callee: esprima.nodes.Node) -> str | None:
        if callee.type == 'Identifier':
            return callee.name
        if callee.type == 'MemberExpression':
            obj = self._get_callee_name(callee.object)
            prop = callee.property.name if callee.property.type == 'Identifier' else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None