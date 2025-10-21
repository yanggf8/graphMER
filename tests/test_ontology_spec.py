from src.ontology.validator import validate_ontology_spec


def test_ontology_spec_structure():
    ok, errors = validate_ontology_spec("docs/specs/ontology_spec.yaml")
    assert ok, "Ontology spec invalid: " + " | ".join(errors)
