from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


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
    sirket_id:       Optional[int] = None
    sube_id:         Optional[int] = None
    mudurluk_id:      Optional[int] = None
    birim_id:        Optional[int] = None
    grup_ids:        list[int] = []
    bina_id:         Optional[int] = None
    pozisyon_id:     Optional[int] = None
    personel_tip_id: Optional[int] = None
    kullanici_id:    Optional[int] = None


def _rule_matches(rule: AudienceRule, user: UserContext) -> bool:
    """AND mantığı: kuraldaki tüm dolu alanlar kullanıcı ile eşleşmeli."""
    if rule.is_empty():
        return False

    if rule.sirket_ids is not None and user.sirket_id not in rule.sirket_ids:
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
