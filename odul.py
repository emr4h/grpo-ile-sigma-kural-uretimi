"""
GRPO ödül fonksiyonları — Sigma kuralı üretimi.

GRPO NASIL ÇALIŞIYOR?
    SFT'de modele doğru cevabı gösterirsiniz. GRPO'da doğru cevap yoktur,
    bir PUANLAYICI vardır: model aynı soruya 8 cevap üretir, hepsi puanlanır,
    model iyi puan alanlara doğru kayar.

    Tek şartı ödülün hesaplanabilir olması. Bu dosya o hesabı yapıyor.

DÖRT BİLEŞEN — ÜÇÜ KESİN
    ayristir  pySigma YAML+şema kabul ediyor mu        kesin   0/1
    donustur  backend condition'ı çözebiliyor mu       kesin   0/1
    attack    ATT&CK kimliği referansla eşleşiyor mu   kesin   0/1
    ortusme   logsource + alan adları ne kadar tuttu   kısmi   0..1

NEDEN İKİ AYRI DOĞRULAMA ADIMI?
    Ölçüldü (16.09.2026): `SigmaCollection.from_yaml()` tek başına yetmiyor.
    Var olmayan bir seçime atıf yapan `condition: olmayan_secim` ayrıştırmadan
    GEÇİYOR. Backend dönüşümü yakalıyor. Tek adıma güvenseydik model
    "geçerli görünen ama çalışmayan" kurallar üretmeyi öğrenebilirdi — tam
    olarak kaçınmaya çalıştığımız şey.

    Yakalananlar:
        bozuk YAML, detection yok, logsource yok, geçersiz değiştirici -> 1. adım
        tanımsız condition referansı, boş detection                    -> 2. adım

ÖDÜLÜ KANDIRMA (reward hacking)
    Model puanı yükseltmenin kestirme yolunu bulursa görevi değil puanlayıcıyı
    öğrenir. Korunma: ödülün ağırlığı kesin bileşenlerde olmalı. Burada
    kesin bileşenler toplam 2.0, kısmi bileşen 0.3 — kısmi olanı büyütmek
    modeli "referansa benzeyen ama çalışmayan" metin üretmeye teşvik ederdi.

KULLANIM
    python odul.py --test        # gerçek kurallarla ve bozuk örneklerle sına
"""
import argparse
import json
import re

import yaml
from pathlib import Path

BURASI = Path(__file__).parent
VERI = BURASI / "veri"

# Ağırlıklar. Kesin olanlar baskın.
#
# A_DONUSTUR neden 1.0? Ölçüldü (16.09.2026): 0.5 iken "tanımsız condition"
# hatası olan bir kural tavanın %78'ini alıyordu (2.30 üzerinden 1.80).
# Çalışmayan bir kural için fazla cömert; model "condition'ı biraz bozuk
# bırak, gerisi tutsun" kestirmesini bulabilirdi. 1.0'de fark 0.5'ten 1.0'e
# çıkıyor.
#
# Denenmeyen alternatif: attack ve ortusme'yi dönüşüme BAĞLAMAK (kural
# çalışmıyorsa etiketinin anlamı yok). Daha keskin ama riskli — eğitimin
# başında üretimlerin çoğu dönüşemez, hepsi aynı puanda toplanır ve GRPO'nun
# karşılaştıracak bir şeyi kalmaz. Ağırlık artırmak sinyali koruyor.
A_AYRISTIR = 0.5
A_DONUSTUR = 1.0
A_ATTACK = 1.0
A_ORTUSME = 0.3

T_KIMLIK = re.compile(r"\bt\d{4}(?:\.\d{3})?\b", re.I)


def yaml_cikar(metin):
    """Model cevabından YAML bloğunu ayıklar.

    Model ```yaml ... ``` çitiyle sarabilir ya da düz yazabilir. İkisini de
    kabul ediyoruz — biçim cezası ayrı bir bileşen değil, çünkü asıl
    ölçtüğümüz şey kuralın geçerliliği.
    """
    m = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", metin, re.S)
    return (m.group(1) if m else metin).strip()


def dogrula(metin):
    """İki adımlı doğrulama. (ayristi, donustu, kural) döndürür."""
    from sigma.collection import SigmaCollection
    try:
        from sigma.backends.splunk import SplunkBackend
        arka = SplunkBackend()
    except Exception:
        arka = None

    ham = yaml_cikar(metin)
    try:
        koleksiyon = SigmaCollection.from_yaml(ham)
    except Exception:
        return False, False, None
    if not koleksiyon.rules:
        return False, False, None

    if arka is None:
        return True, False, koleksiyon.rules[0]
    try:
        arka.convert(koleksiyon)
    except Exception:
        return True, False, koleksiyon.rules[0]
    return True, True, koleksiyon.rules[0]


def _attack(kural):
    return sorted({m.group(0).lower()
                   for t in (getattr(kural, "tags", None) or [])
                   for m in [T_KIMLIK.search(str(t))] if m})


def _taktikler(kural):
    """tags içindeki taktik etiketleri (attack.execution gibi)."""
    return {str(t).lower().replace("attack.", "")
            for t in (getattr(kural, "tags", None) or [])
            if str(t).lower().startswith("attack.")
            and not T_KIMLIK.search(str(t))}


def attack_puani(bulunan, beklenen, bulunan_taktik=(), beklenen_taktik=()):
    """ATT&CK isabetini KADEMELİ puanlar. 0..1 arası.

    NEDEN İKİLİ DEĞİL? Ölçüldü (16.09.2026) ve GRPO'yu çökertti.
    İkili ödülde (tam isabet 1.0, yanlış 0) 390 teknik arasından doğruyu
    bulma şansı %0.26. Sekiz üretimin hepsi 0 alıyor, `std = 0` oluyor ve
    GRPO o bileşenden hiçbir yön bilgisi çıkaramıyor. Geriye kalan gradyan
    yalnızca biçim bileşenlerinden geliyordu; model orada zaten %87'deydi,
    kazanacak yeri yoktu. Sonuç: rastgele dolaşma ve SFT'de öğrenilen
    biçimin erozyonu. 20 adımda ödül 1.575'ten 1.327'ye düştü.

    Kademeli puan modelin TIRMANMASINI sağlıyor: önce doğru taktiği bulur,
    sonra teknik ailesini, sonra tam kimliği. Her basamakta std > 0.

        t1059.001 = t1059.001   tam isabet        1.0
        t1059     ~ t1059.001   doğru aile        0.5
        execution ~ execution   doğru taktik      0.2
        alakasız                                  0.0
    """
    bulunan, beklenen = set(bulunan or []), set(beklenen or [])
    if not beklenen:
        return 0.0
    if bulunan & beklenen:
        return 1.0
    # Aile eşleşmesi: t1059.001 -> t1059
    aile_b = {x.split(".")[0] for x in bulunan}
    aile_k = {x.split(".")[0] for x in beklenen}
    if aile_b & aile_k:
        return 0.5
    if set(bulunan_taktik or []) & set(beklenen_taktik or []):
        return 0.2
    return 0.0


def _alanlar(detection):
    """detection içindeki alan adlarını toplar (değiştiriciler atılır)."""
    out = set()

    def gez(x):
        if isinstance(x, dict):
            for a, v in x.items():
                if a != "condition":
                    out.add(str(a).split("|")[0].lower())
                gez(v)
        elif isinstance(x, list):
            for v in x:
                gez(v)
    gez(detection)
    return out


def puanla(cevap, referans):
    """Tek bir cevabı puanlar. referans: havuzdaki kayıt (dict)."""
    p = {"ayristir": 0.0, "donustur": 0.0, "attack": 0.0, "ortusme": 0.0}
    ayristi, donustu, kural = dogrula(cevap)
    p["ayristir"] = A_AYRISTIR if ayristi else 0.0
    p["donustur"] = A_DONUSTUR if donustu else 0.0
    if not ayristi:
        return p, "ayrıştırılamadı"

    # KESİN ama KADEMELİ: ATT&CK kimliği ne kadar tuttu?
    p["attack"] = A_ATTACK * attack_puani(
        _attack(kural), referans.get("attack") or [],
        _taktikler(kural), referans.get("taktik") or [])

    # KISMİ: logsource ve alan adları ne kadar tuttu?
    ls_ref = referans.get("logsource") or {}
    ls_kural = getattr(kural, "logsource", None)
    ls_puan = 0.0
    if ls_kural is not None:
        eslesen = sum(1 for a in ("product", "category", "service")
                      if ls_ref.get(a)
                      and str(getattr(ls_kural, a, "") or "").lower() == str(ls_ref[a]).lower())
        toplam = sum(1 for a in ("product", "category", "service") if ls_ref.get(a))
        ls_puan = eslesen / toplam if toplam else 0.0

    ref_alan = _alanlar(referans.get("detection") or {})
    try:
        cik_alan = _alanlar((yaml.safe_load(yaml_cikar(cevap)) or {}).get("detection") or {})
    except Exception:
        cik_alan = set()
    alan_puan = len(ref_alan & cik_alan) / len(ref_alan) if ref_alan else 0.0

    p["ortusme"] = A_ORTUSME * (ls_puan + alan_puan) / 2
    return p, ""


# --- TRL arayüzü ---------------------------------------------------------
# GRPOTrainer'a verilen ödül fonksiyonları `grpo_kos.py` içinde tanımlı,
# burada DEĞİL. Sebebi: TRL onlara veri setindeki sütunları kwargs olarak
# geçiriyor (attack, taktik, ls_product...) ve o şema eğitim tarafına ait.
#
# Burada bir kopya tutmayı denedik ve hata oldu: kademeli puana geçince
# `grpo_kos.py` güncellendi, buradaki kopya İKİLİ kaldı. Kullanılmıyordu
# ama okuyanı yanıltıyordu. Tek kaynak ilkesi — `puanla()` ve
# `attack_puani()` burada, TRL sarmalayıcıları orada.

def test():
    """Eğitimden ÖNCE ödülü sına. Yanlış ödülle üç saat eğitim koşup sonunda
    modelin puanlayıcıyı kandırdığını görmek istemiyoruz."""
    kayitlar = [json.loads(s) for s in
                open(VERI / "havuz_test.jsonl", encoding="utf-8") if s.strip()][:3]

    print("=== GERÇEK KURALLAR (tam puan almalı) ===")
    for r in kayitlar:
        p, n = puanla(r["ham"], r)
        print(f"  {sum(p.values()):.2f}  {p}  {r['kaynak'].split('/')[-1][:40]}")

    ref = kayitlar[0]
    iyi = ref["ham"]
    print("\n=== BOZUK ÇEŞİTLERİ (düşük puan almalı) ===")
    bozuklar = {
        "bozuk YAML": iyi.replace("detection:", "detection:::"),
        "detection yok": iyi.replace("detection:", "xdetection:"),
        "logsource yok": iyi.replace("logsource:", "xlogsource:"),
        "tanımsız condition": re.sub(r"condition: .*", "condition: yok_boyle_bir_sey", iyi),
        "yanlış ATT&CK": re.sub(r"attack\.t\d{4}(\.\d{3})?", "attack.t9999", iyi),
        "boş cevap": "",
        "sadece metin": "Bu bir Sigma kuralı değil, sadece açıklama cümlesi.",
    }
    for ad, metin in bozuklar.items():
        p, n = puanla(metin, ref)
        print(f"  {sum(p.values()):.2f}  {ad:<20} {p}")

    # --- AYIRIM TESTİ ------------------------------------------------------
    # Referansı kendisiyle puanlamak yalnızca TAVANI ölçer. Ödülün işe
    # yaradığını görmek için yanlış ama GEÇERLİ cevapların ne aldığına
    # bakmak gerekiyor: model rastgele düzgün bir kural üreterek yüksek
    # puan alabiliyorsa ödül bozuktur.
    import statistics as st
    hepsi = [json.loads(s) for s in
             open(VERI / "havuz_test.jsonl", encoding="utf-8") if s.strip()]
    n = min(25, len(hepsi))

    dogru = [sum(puanla(r["ham"], r)[0].values()) for r in hepsi[:n]]
    yanlis = [sum(puanla(hepsi[(i + 7) % len(hepsi)]["ham"], r)[0].values())
              for i, r in enumerate(hepsi[:n])]

    # En zor ayrım: aynı ATT&CK etiketli başka bir kural.
    from collections import defaultdict
    gruplar = defaultdict(list)
    for r in hepsi:
        for t in r["attack"]:
            gruplar[t].append(r)
    ayni = [sum(puanla(g[1]["ham"], g[0])[0].values())
            for g in gruplar.values() if len(g) >= 2]

    print(f"\n=== AYIRIM ({n} kayıt) ===")
    print(f"  doğru cevap                : {st.mean(dogru):.2f}")
    print(f"  alakasız ama geçerli kural : {st.mean(yanlis):.2f}")
    if ayni:
        print(f"  aynı ATT&CK, başka kural   : {st.mean(ayni):.2f}  (n={len(ayni)})")
    print(f"  AYIRIM GÜCÜ                : {st.mean(dogru)-st.mean(yanlis):.2f}")

    print("\n  NASIL OKUNUR")
    print("  Gerçek kurallar tavanı almalı (dört bileşen de dolu).")
    print("  'yanlış ATT&CK' ayrıştırma ve dönüşümü geçip attack bileşenini")
    print("  kaybetmeli; 'tanımsız condition' ayrıştırmayı geçip dönüşümde")
    print("  takılmalı. İki satır da kesin bileşenlerin ayırt ettiğini gösterir.")
    print("  Ayırım gücü 1.0'ın altına düşerse ödül zayıflamış demektir.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    test() if a.test else ap.print_help()
