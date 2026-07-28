"""
tests/test_abac_tenant_scoping.py — v2.0 mimarisinde sirket_ids'in artık
tenant sınırından bağımsız, sıradan bir ÇOĞUL hedef kitle özniteliği olarak
ABAC filtresine TAM katıldığını doğrular (musteri_id fiziksel tenant sınırı
olduğundan sirket_ids hiçbir zaman filtre dışı bırakılmaz). exclude_fields
mekanizması genel bir uzatma noktası olarak (örn. yaka_tipi_ids ile) ayrıca
test edilir.
"""
from abac import UserContext, AudienceRule, AudiencePolicy, has_access, build_qdrant_abac_filter


class TestSirketIdsFullyParticipatesInFilter:
    """(v2.0) sirket_ids artık koleksiyon içinde anlamsız/yedek DEĞİLDİR —
    bir müşterinin birden fazla şirketi olabileceğinden gerçek bir erişim
    kısıtlamasıdır ve varsayılan filtreye her zaman dahildir."""

    def test_default_filter_includes_sirket_ids_condition(self):
        user = UserContext(musteri_id=501, sirket_ids=[14], bina_id=16)
        filt = build_qdrant_abac_filter(user)
        assert "sirket_ids" in str(filt)

    def test_sirket_ids_uses_match_any_like_grup_ids(self):
        user = UserContext(musteri_id=501, sirket_ids=[14, 18])
        filt = build_qdrant_abac_filter(user)
        filt_str = str(filt)
        assert "MatchAny" in filt_str

    def test_existing_has_access_logic_untouched_for_other_fields(self):
        user = UserContext(musteri_id=501, sirket_ids=[14], bina_id=16, personel_tip_id=1)
        policy = AudiencePolicy(rules=[
            AudienceRule(bina_ids=[16], personel_tip_ids=[1]),
            AudienceRule(bina_ids=[17]),
        ])
        assert has_access(policy, user) is True


class TestSirketIdsIntersectionMatching:
    """(v2.0) Bir kullanıcı birden fazla şirkete bağlı olabilir; kesişimde
    EN AZ BİR ortak şirket varsa kural eşleşir."""

    def test_matches_when_any_sirket_overlaps(self):
        user = UserContext(musteri_id=501, sirket_ids=[14, 18])
        policy = AudiencePolicy(rules=[AudienceRule(sirket_ids=[18, 23])])
        assert has_access(policy, user) is True

    def test_does_not_match_when_no_overlap(self):
        user = UserContext(musteri_id=501, sirket_ids=[14])
        policy = AudiencePolicy(rules=[AudienceRule(sirket_ids=[18, 23])])
        assert has_access(policy, user) is False

    def test_empty_sirket_ids_only_matches_wildcard_rules(self):
        user = UserContext(musteri_id=501, sirket_ids=[])
        policy_restricted = AudiencePolicy(rules=[AudienceRule(sirket_ids=[14])])
        policy_wildcard = AudiencePolicy(rules=[AudienceRule(bina_ids=[16])])

        assert has_access(policy_restricted, user) is False
        user_with_bina = UserContext(musteri_id=501, sirket_ids=[], bina_id=16)
        assert has_access(policy_wildcard, user_with_bina) is True


class TestExcludeFieldsGeneralPurposeMechanism:
    """exclude_fields, musteri_id ile ilgisi olmayan genel bir uzatma
    noktası olarak korunur (OCP) — herhangi bir alan için çalışır."""

    def test_yaka_tipi_can_be_excluded(self):
        user = UserContext(musteri_id=501, sirket_ids=[14], yaka_tipi_id=3)
        filt = build_qdrant_abac_filter(user, exclude_fields=frozenset({"yaka_tipi_ids"}))
        assert "yaka_tipi_ids" not in str(filt)

    def test_sirket_ids_can_still_be_excluded_if_explicitly_requested(self):
        user = UserContext(musteri_id=501, sirket_ids=[14])
        filt = build_qdrant_abac_filter(user, exclude_fields=frozenset({"sirket_ids"}))
        assert "sirket_ids" not in str(filt)