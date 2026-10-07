import pytest

from discrete import textkit as tk
from discrete.process import entities


@pytest.mark.parametrize("value", ["false", "true", 1, [], None])
async def test_entity_adjudication_requires_boolean_true(monkeypatch, fake, value):
    async def response(*args, **kwargs):
        return {"same": value}

    monkeypatch.setattr(entities, "complete_json", response)
    assert await entities._adjudicate("Acme", "Acne") is False


async def test_entity_adjudication_accepts_true(monkeypatch, fake):
    async def response(*args, **kwargs):
        return {"same": True}

    monkeypatch.setattr(entities, "complete_json", response)
    assert await entities._adjudicate("Acme", "Acne") is True


@pytest.mark.parametrize("existing,name", [("acme.com", "acme.org"),
                                           ("Project Atlas Production 101", "Project Atlas Production 102"),
                                           ("projectatlasproduction101", "projectatlasproduction102")])
async def test_different_identifiers_are_not_merged(store, existing, name):
    await store.create_entity(existing, entities.infer_kind(existing), tk.entity_key(existing))
    assert await entities.resolve_existing(store, name) is None


async def test_a_name_still_meets_its_domain(store):
    eid = await store.create_entity("acme.com", "website", tk.entity_key("acme.com"))
    assert await entities.resolve_existing(store, "Acme") == eid
