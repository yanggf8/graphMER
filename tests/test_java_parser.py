"""Test Java parser basic functionality"""
from src.parsing.java_parser import JavaParser

def test_java_parser_basic():
    parser = JavaParser()
    code = """
import java.util.List;

public class Test {
    public void method() {
        helper();
    }
}
"""
    triples = parser.parse(code, "Test.java")
    
    # Should find class, method, import, and call
    assert len(triples) >= 3
    
    # Check for class definition
    class_triples = [t for t in triples if t['head']['type'] == 'class']
    assert len(class_triples) >= 1
    
    # Check for import
    import_triples = [t for t in triples if t['relation']['type'] == 'imports']
    assert len(import_triples) >= 1
