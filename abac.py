"""
abac.py — ABAC (Attribute-Based Access Control) çekirdek modeli.

TASARIM KARARI — Müşteri Seviyesinde İzolasyon (v2.0):
──────────────────────────────────────────────────────────────────────────────
Önceki tasarımda sirket_id (şirket kimliği) tenant/izolasyon sınırı olarak
kullanılıyordu. Teracity'nin bulgusu: Bilimp'in HER MÜŞTERİSİ kendi ayrı
veritabanına sahip ve şirket kimlikleri yalnızca o müşterinin kendi
veritabanı içinde üretiliyor — yani global olarak BENZERSİZ DEĞİL. İki farklı
Bilimp müşterisinde aynı sirket_id (hatta aynı kullanici_id) rastlantısal
olarak çakışabilir. sirket_id'yi tenant sınırı olarak kullanmak, iki farklı
müşterinin verisinin aynı fiziksel alanda karışmasına yol açardı.

Bu yüzden:
  - musteri_id: Bilimp'in ürettiği, GLOBAL BENZERSİZ müşteri (hesap) kimliği.
    Artık gerçek tenant/izolasyon sınırı budur (bkz. tenancy.py). UserContext'te
    ZORUNLU bir alandır — bir kullanıcı bağlamı, hangi müşteriye ait olduğu
    bilinmeden anlamsızdır.
  - sirket_id → sirket_ids (ÇOĞUL): Artık bir izolasyon anahtarı değil, şube/
    bina gibi SIRADAN bir hedef kitle özniteliğidir; bir müşterinin birden
    fazla şirketi olabileceği gibi, bir kullanıcı da birden fazla şirkete
    bağlı olabilir — bu yüzden grup_ids ile aynı çoğul/kesişim modelini
    kullanır.

AudienceRule'da musteri_id İÇİN bir alan YOKTUR ve olmayacaktır: bir kural
zaten yalnızca kendi müşterisinin fiziksel koleksiyonu içinde değerlendirilir
(tenancy.py); çapraz-müşteri bir kural kavramı sistemde hiç yoktur.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


FIELD_LABELS = {
    "sirket_ids":       "Şirket",
    "sube_ids":         "Şube",
    "mudurluk_ids":      "Müdürlük",
    "birim_ids":        "Birim",
    "grup_ids":         "Grup",
    "bina_ids":         "Bina",
    "pozisyon_ids":     "Pozisyon",
    "personel_tip_ids": "Personel Tipi",
    "kullanici_ids":    "Kullanıcı",
    "yaka_tipi_ids":    "Yaka Tipi",
}


class AudienceRule(BaseModel):
    sirket_ids:       Optional[list[int]] = None
    sube_ids:         Optional[list[int]] = None
    mudurluk_ids:      Optional[list[int]] = None
    birim_ids:        Optional[list[int]] = None
    grup_ids:         Optional[list[int]] = None
    bina_ids:         Optional[list[int]] = None
    pozisyon_ids:     Optional[list[int]] = None
    personel_tip_ids: Optional[list[int]] = None
    kullanici_ids:    Optional[list[int]] = None
    yaka_tipi_ids:    Optional[list[int]] = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())

    def summary(self) -> str:
        parts = []
        data = self.model_dump()
        for field, label in FIELD_LABELS.items():
            val = data.get(field)
            if val:
                parts.append(f"{label}:{val}")
        return ", ".join(parts) if parts else "—"


class AudiencePolicy(BaseModel):
    rules: list[AudienceRule] = []

    def is_empty(self) -> bool:
        return len(self.rules) == 0 or all(r.is_empty() for r in self.rules)

    def summary(self) -> str:
        if self.is_empty():
            return "Erişim Yok"
        parts = [f"[{r.summary()}]" for r in self.rules if not r.is_empty()]
        return " VEYA ".join(parts) if parts else "Erişim Yok"


class UserContext(BaseModel):
    # (v2.0) Bilimp müşteri (hesap) kimliği — global benzersiz, ZORUNLU.
    # Fiziksel tenant sınırı budur; bkz. tenancy.py.
    musteri_id: int = Field(..., description="Bilimp müşteri (hesap) kimliği — global benzersiz")

    # (v2.0) Artık izolasyon anahtarı DEĞİL, sıradan bir öznitelik — bir
    # kullanıcı birden fazla şirkete bağlı olabileceğinden grup_ids ile aynı
    # çoğul/kesişim modelini kullanır.
    sirket_ids:      list[int] = []

    sube_id:         Optional[int] = None
    mudurluk_id:      Optional[int] = None
    birim_id:        Optional[int] = None
    grup_ids:        list[int] = []
    bina_id:         Optional[int] = None
    pozisyon_id:     Optional[int] = None
    personel_tip_id: Optional[int] = None
    kullanici_id:    Optional[int] = None
    yaka_tipi_id:    Optional[int] = None


def _rule_matches(rule: AudienceRule, user: UserContext) -> bool:
    """AND mantığı: kuraldaki tüm dolu alanlar kullanıcı ile eşleşmeli."""
    if rule.is_empty():
        return False

    # (v2.0) sirket_ids artık grup_ids ile aynı ÇOĞUL/KESİŞİM modelini
    # kullanır: kuralın istediği şirketlerden EN AZ BİRİ kullanıcının
    # bağlı olduğu şirketler arasında olmalı.
    if rule.sirket_ids is not None and not any(s in rule.sirket_ids for s in user.sirket_ids):
        return False
    if rule.sube_ids is not None and user.sube_id not in rule.sube_ids:
        return False
    if rule.mudurluk_ids is not None and user.mudurluk_id not in rule.mudurluk_ids:
        return False
    if rule.birim_ids is not None and user.birim_id not in rule.birim_ids:
        return False
    if rule.grup_ids is not None and not any(g in rule.grup_ids for g in user.grup_ids):
        return False
    if rule.bina_ids is not None and user.bina_id not in rule.bina_ids:
        return False
    if rule.pozisyon_ids is not None and user.pozisyon_id not in rule.pozisyon_ids:
        return False
    if rule.personel_tip_ids is not None and user.personel_tip_id not in rule.personel_tip_ids:
        return False
    if rule.kullanici_ids is not None and user.kullanici_id not in rule.kullanici_ids:
        return False
    if rule.yaka_tipi_ids is not None and user.yaka_tipi_id not in rule.yaka_tipi_ids:
        return False

    return True


def has_access(policy: AudiencePolicy, user: UserContext) -> bool:
    """OR mantığı: en az bir kural eşleşirse erişim verilir. Boş policy = erişim yok."""
    if policy.is_empty():
        return False
    return any(_rule_matches(rule, user) for rule in policy.rules)


def parse_ids(text: str) -> Optional[list[int]]:
    """'14, 18, 23' → [14, 18, 23]. Boş veya geçersiz → None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        ids = [int(x.strip()) for x in text.split(",") if x.strip()]
        return ids if ids else None
    except ValueError:
        return None


def build_policy_from_ui(rules_data: list[dict]) -> AudiencePolicy:
    """Streamlit form verisinden AudiencePolicy oluşturur."""
    rules = []
    for rd in rules_data:
        kwargs = {}
        for field in FIELD_LABELS:
            parsed = parse_ids(rd.get(field, ""))
            if parsed is not None:
                kwargs[field] = parsed
        if kwargs:
            rules.append(AudienceRule(**kwargs))
    return AudiencePolicy(rules=rules)


def empty_rule_data() -> dict:
    return {f: "" for f in FIELD_LABELS}


# ── Qdrant Pre-Filter ─────────────────────────────────────────────────────────

# (v2.0) Çoklu değerli öznitelikler — kullanıcı tarafında bir LİSTE olarak
# tutulan, kesişim mantığıyla eşleşen alanlar. sirket_ids'in grup_ids ile
# aynı listede olması bilinçlidir: DRY — iki alan birbirinin kopyası mantığı
# paylaşır, tekrar kod yazılmaz.
_MULTI_VALUED_FIELDS = ("sirket_ids", "grup_ids")


def build_qdrant_abac_filter(user: UserContext, exclude_fields: frozenset[str] = frozenset()):
    """
    UserContext'e göre Qdrant veritabanı seviyesinde pre-filter üretir.

    Mantık:
      metadata.audience.rules dizisindeki herhangi bir kural (OR)
      kullanıcının öznitelikleriyle tam eşleşirse belge döndürülür.

      Kural içi: AND — kuraldaki her dolu alan eşleşmeli.
      Boş alan (null / []) = wildcard — o öznitelikte kısıtlama yok.
      Çoklu değerli alanlar (sirket_ids, grup_ids): kesişim — kullanıcının
      listesindeki değerlerden EN AZ BİRİ kuralın listesinde olmalı.

    Avantaj:
      Python post-filter yerine Qdrant motoru filtrelediği için
      gerçekten erişimi olan k belge doğrudan döner.

    exclude_fields:
      Belirtilen alan(lar) filtreye hiç dahil edilmez. musteri_id (fiziksel
      tenant sınırı, bkz. tenancy.py) hiçbir zaman bir AudienceRule alanı
      olmadığından bu parametreyle ilişkili değildir; bu genel bir uzatma
      noktası olarak korunur (OCP).
    """
    from qdrant_client.http import models as qm

    must_conditions: list = []

    # ── Tekil değerli öznitelikler ────────────────────────────────────────────
    single_attrs = [
        ("sube_ids",         user.sube_id),
        ("mudurluk_ids",     user.mudurluk_id),
        ("birim_ids",        user.birim_id),
        ("bina_ids",         user.bina_id),
        ("pozisyon_ids",     user.pozisyon_id),
        ("personel_tip_ids", user.personel_tip_id),
        ("kullanici_ids",    user.kullanici_id),
        ("yaka_tipi_ids",    user.yaka_tipi_id),
    ]

    for field_key, user_val in single_attrs:
        if field_key in exclude_fields:
            continue
        if user_val is None:
            # Kullanıcının bu özniteliği yok → yalnızca bu alanı kısıtlamayan
            # (wildcard/boş) kurallara erişebilir.
            must_conditions.append(
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key=field_key))
            )
        else:
            # Kural bu alanı boş bırakmış (wildcard) VEYA kullanıcı değeri listede
            must_conditions.append(
                qm.Filter(should=[
                    qm.IsEmptyCondition(is_empty=qm.PayloadField(key=field_key)),
                    qm.FieldCondition(key=field_key, match=qm.MatchValue(value=user_val)),
                ])
            )

    # ── Çoklu değerli öznitelikler: sirket_ids, grup_ids (kesişim) ───────────
    multi_attrs = [(f, getattr(user, f)) for f in _MULTI_VALUED_FIELDS]
    for field_key, user_vals in multi_attrs:
        if field_key in exclude_fields:
            continue
        if user_vals:
            # Kural bu alanı istemiyorsa (wildcard) VEYA kullanıcının
            # listesinden biri kural listesinde
            must_conditions.append(
                qm.Filter(should=[
                    qm.IsEmptyCondition(is_empty=qm.PayloadField(key=field_key)),
                    qm.FieldCondition(key=field_key, match=qm.MatchAny(any=user_vals)),
                ])
            )
        else:
            # Kullanıcının bu öznitelikte hiç değeri yoksa — yalnızca bu
            # alanda kısıtlama olmayan kurallara erişim
            must_conditions.append(
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key=field_key))
            )

    # ── NestedCondition: rules dizisinde EN AZ BİR kural eşleşmeli ───────────
    return qm.Filter(
        must=[
            qm.NestedCondition(
                nested=qm.Nested(
                    key="metadata.audience.rules",
                    filter=qm.Filter(must=must_conditions),
                )
            )
        ]
    )