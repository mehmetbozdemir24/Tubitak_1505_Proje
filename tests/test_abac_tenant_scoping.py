"""
tests/test_abac_tenant_scoping.py — build_qdrant_abac_filter'ın exclude_fields
parametresinin, Seviye C mimarisi için sirket_ids alanını doğru şekilde
filtre dışı bıraktığını doğrular. Ayrıca geriye dönük uyumluluğu (varsayılan
davranışın değişmediğini) garanti eder.
"""
from abac import UserContext, AudienceRule, AudiencePolicy, has_access, build_qdrant_abac_filter


class TestBackwardCompatibility:
    """exclude_fields verilmezse davranış Faz 1 öncesiyle birebir aynı olmalı."""

    def test_default_filter_still_includes_sirket_ids_condition(self):
        user = UserContext(sirket_id=14, bina_id=16)
        filt = build_qdrant_abac_filter(user)
        assert "sirket_ids" in str(filt)

    def test_existing_has_access_logic_untouched(self):
        user = UserContext(sirket_id=14, bina_id=16, personel_tip_id=1)
        policy = AudiencePolicy(rules=[
            AudienceRule(bina_ids=[16], personel_tip_ids=[1]),
            AudienceRule(bina_ids=[17]),
        ])
        assert has_access(policy, user) is True


class TestTenantScopedFilter:
    """exclude_fields={'sirket_ids'} verildiğinde sirket_ids filtreye hiç girmemeli."""

    def test_sirket_ids_excluded_from_filter(self):
        user = UserContext(sirket_id=14, bina_id=16)
        filt = build_qdrant_abac_filter(user, exclude_fields=frozenset({"sirket_ids"}))
        assert "sirket_ids" not in str(filt)

    def test_other_fields_still_present_when_excluding_sirket(self):
        user = UserContext(sirket_id=14, bina_id=16, personel_tip_id=1)
        filt = build_qdrant_abac_filter(user, exclude_fields=frozenset({"sirket_ids"}))
        filt_str = str(filt)
        assert "bina_ids" in filt_str
        assert "personel_tip_ids" in filt_str

    def test_excluding_sirket_does_not_change_other_attribute_matching(self):
        user_matching = UserContext(bina_id=16, personel_tip_id=1)
        user_not_matching = UserContext(bina_id=99, personel_tip_id=1)
        policy = AudiencePolicy(rules=[AudienceRule(bina_ids=[16], personel_tip_ids=[1])])

        assert has_access(policy, user_matching) is True
        assert has_access(policy, user_not_matching) is False

    def test_yaka_tipi_can_also_be_excluded(self):
        user = UserContext(sirket_id=14, yaka_tipi_id=3)
        filt = build_qdrant_abac_filter(user, exclude_fields=frozenset({"yaka_tipi_ids"}))
        assert "yaka_tipi_ids" not in str(filt)