"""
tests/test_tenancy.py — tenancy.py için birim testleri.

Gerçek bir Qdrant sunucusu gerektirmez; QdrantClient sahte (mock) nesnelerle
temsil edilir.
"""
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from tenancy import (
    ConventionTenantRegistry,
    TenantCollectionProvisioner,
    resolve_tenant_collection,
)


# ══════════════════════════════════════════════════════════════════════════════
# ConventionTenantRegistry
# ══════════════════════════════════════════════════════════════════════════════
class TestConventionTenantRegistry:
    def test_collection_name_deterministic(self):
        registry = ConventionTenantRegistry()
        assert registry.collection_name(14) == "tubitak1505_musteri_14"
        assert registry.collection_name(18) == "tubitak1505_musteri_18"

    def test_different_tenants_get_different_collections(self):
        registry = ConventionTenantRegistry()
        assert registry.collection_name(14) != registry.collection_name(18)

    def test_same_tenant_always_same_collection(self):
        registry = ConventionTenantRegistry()
        assert registry.collection_name(14) == registry.collection_name(14)

    def test_rejects_negative_tenant_id(self):
        registry = ConventionTenantRegistry()
        with pytest.raises(ValueError):
            registry.collection_name(-1)

    def test_rejects_none_tenant_id(self):
        registry = ConventionTenantRegistry()
        with pytest.raises(ValueError):
            registry.collection_name(None)

    def test_reverse_resolution(self):
        registry = ConventionTenantRegistry()
        assert registry.tenant_id_from_collection("tubitak1505_musteri_14") == 14

    def test_reverse_resolution_unknown_prefix_returns_none(self):
        registry = ConventionTenantRegistry()
        assert registry.tenant_id_from_collection("baska_koleksiyon") is None

    def test_reverse_resolution_malformed_suffix_returns_none(self):
        registry = ConventionTenantRegistry()
        assert registry.tenant_id_from_collection("tubitak1505_musteri_abc") is None


# ══════════════════════════════════════════════════════════════════════════════
# resolve_tenant_collection
# ══════════════════════════════════════════════════════════════════════════════
class TestResolveTenantCollection:
    def test_resolves_with_valid_musteri_id(self):
        registry = ConventionTenantRegistry()
        assert resolve_tenant_collection(registry, 14) == "tubitak1505_musteri_14"

    def test_raises_400_when_musteri_id_missing(self):
        registry = ConventionTenantRegistry()
        with pytest.raises(HTTPException) as exc_info:
            resolve_tenant_collection(registry, None)
        assert exc_info.value.status_code == 400

    def test_raises_400_on_invalid_musteri_id(self):
        registry = ConventionTenantRegistry()
        with pytest.raises(HTTPException) as exc_info:
            resolve_tenant_collection(registry, -5)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# TenantCollectionProvisioner
# ══════════════════════════════════════════════════════════════════════════════
class TestTenantCollectionProvisioner:
    def test_returns_false_when_collection_already_exists(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        provisioner = TenantCollectionProvisioner(client, reference_collection="ref")

        created = provisioner.ensure_exists("tubitak1505_musteri_14")

        assert created is False
        client.create_collection.assert_not_called()

    def test_creates_collection_copying_reference_schema(self):
        client = MagicMock()
        # İlk çağrı (hedef koleksiyon var mı) -> False, ikinci (referans var mı) -> True
        client.collection_exists.side_effect = [False, True]

        fake_ref = MagicMock()
        fake_ref.config.params.vectors = {"content": "FAKE_DENSE_CONFIG"}
        fake_ref.config.params.sparse_vectors = {"sparse": "FAKE_SPARSE_CONFIG"}
        client.get_collection.return_value = fake_ref

        provisioner = TenantCollectionProvisioner(client, reference_collection="ref")
        created = provisioner.ensure_exists("tubitak1505_musteri_18")

        assert created is True
        client.create_collection.assert_called_once_with(
            collection_name="tubitak1505_musteri_18",
            vectors_config={"content": "FAKE_DENSE_CONFIG"},
            sparse_vectors_config={"sparse": "FAKE_SPARSE_CONFIG"},
        )

    def test_raises_500_when_reference_collection_missing(self):
        client = MagicMock()
        client.collection_exists.side_effect = [False, False]  # hedef yok, referans da yok
        provisioner = TenantCollectionProvisioner(client, reference_collection="ref")

        with pytest.raises(HTTPException) as exc_info:
            provisioner.ensure_exists("tubitak1505_musteri_99")

        assert exc_info.value.status_code == 500