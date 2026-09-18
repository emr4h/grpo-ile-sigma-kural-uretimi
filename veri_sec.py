"""
SigmaHQ kurallarından havuzu çıkarır.

KAYNAK: github.com/SigmaHQ/sigma — Detection Rule License (DRL) 1.1
    Kullan, değiştir, yayınla serbest; şart atıf (`author` alanı ve kaynağa
    link korunacak). NC/ND yok.

NEDEN SIGMA?
    GRPO'nun tek şartı ödülün hesaplanabilir olması. Sigma'nın çevrimdışı
    doğrulayıcısı var (pySigma), Splunk SPL'in yok — SPL için çalışan bir
    Splunk örneği gerekiyor ve ödül fonksiyonu eğitim boyunca binlerce kez
    çalışıyor. Gerekçenin tamamı README.md içinde.

ÜÇLÜ AYNI DOSYADA
    description → modele verilecek SORU
    detection   → modelin üreteceği CEVAP
    tags        → cevabı DOĞRULAYACAĞIMIZ ATT&CK kimliği
    Yani veri üretmiyoruz; soru da cevap da doğrulayıcı da hazır.

KULLANIM
    python veri_sec.py                 # depoyu çeker, havuzu kurar
    python veri_sec.py --kaynak /tmp/sigma   # zaten indirilmişse
"""
import argparse
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

BURASI = Path(__file__).parent
VERI = BURASI / "veri"

DEPO = "https://github.com/SigmaHQ/sigma.git"

# Yalnızca `rules/` — kararlı kural seti. Diğer klasörler (emerging-threats,
# threat-hunting, placeholder) farklı olgunlukta; karıştırmak ölçümü bulandırır.
KLASOR = "rules"

# Uzunluk sınırları (ham YAML, karakter).
# Üst sınır aykırı değerler için: en büyük kural 295.078 karakter ve aslında
# kural değil, zafiyetli sürücü listesi. 3000 sınırı 2798'den 2687 bırakıyor.
MIN_UZUNLUK = 300
MAX_UZUNLUK = 3000

T_KIMLIK = re.compile(r"^attack\.(t\d{4}(?:\.\d{3})?)$", re.I)


def indir(hedef):
    if (hedef / KLASOR).is_dir():
        print(f"  {hedef} zaten var, indirme atlanıyor")
        return
    print(f"  {DEPO} çekiliyor (sığ klon)...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "-q", DEPO, str(hedef)],
                   check=True)


def kimlikler(kural):
    """tags içinden ATT&CK teknik kimliklerini çıkarır (t1059.001 gibi)."""
    out = []
    for t in (kural.get("tags") or []):
        m = T_KIMLIK.match(str(t))
        if m:
            out.append(m.group(1).lower())
    return sorted(set(out))


def zorluk(kural):
    """condition ifadesinin karmaşıklığı. Test setini buna göre bölmek
    şart: model kolay olanlarda iyi görünüp zorlarda çuvallayabilir."""
    c = str((kural.get("detection") or {}).get("condition", ""))
    if any(x in c for x in (" and ", " or ", " not ")):
        return "mantiksal"
    if " of " in c:
        return "toplu"
    return "tek"


def sec(kaynak):
    import yaml
    kok = Path(kaynak) / KLASOR
    dosyalar = sorted(kok.rglob("*.yml"))
    print(f"  ham dosya            : {len(dosyalar):,}")

    kurallar, hatali = [], 0
    for f in dosyalar:
        try:
            ham = f.read_text(encoding="utf-8")
            d = yaml.safe_load(ham)
        except Exception:
            hatali += 1
            continue
        if not isinstance(d, dict) or "detection" not in d:
            continue
        d["_ham"] = ham
        d["_yol"] = str(f.relative_to(kaynak))
        kurallar.append(d)
    print(f"  ayrıştırılan         : {len(kurallar):,}  (hatalı {hatali})")

    ile_kimlik = [k for k in kurallar if kimlikler(k)]
    print(f"  ATT&CK kimliği olan  : {len(ile_kimlik):,}")

    ile_aciklama = [k for k in ile_kimlik if (k.get("description") or "").strip()]
    print(f"  açıklaması olan      : {len(ile_aciklama):,}")

    uygun = [k for k in ile_aciklama
             if MIN_UZUNLUK <= len(k["_ham"]) <= MAX_UZUNLUK]
    print(f"  uzunluk filtresinden : {len(uygun):,}  "
          f"({MIN_UZUNLUK}-{MAX_UZUNLUK} karakter)")
    return uygun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", default="/tmp/sigma")
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--tohum", type=int, default=42)
    args = ap.parse_args()

    VERI.mkdir(exist_ok=True)
    print("=== KAYNAK ===")
    indir(Path(args.kaynak))

    print("\n=== SEÇİM ===")
    kurallar = sec(args.kaynak)

    # Test setini İKİ eksende dengele: logsource ailesi ve condition zorluğu.
    # Veri Windows'a yığılmış (process_creation tek başına ~1070 kural); düz
    # rastgele bölme test setini oraya kaydırır ve Linux/macOS'taki zayıflığı
    # gizler. Aynısı zorluk için de geçerli.
    r = random.Random(args.tohum)
    gruplar = {}
    for k in kurallar:
        ls = k.get("logsource") or {}
        anahtar = (str(ls.get("product") or "-"), zorluk(k))
        gruplar.setdefault(anahtar, []).append(k)
    for g in gruplar.values():
        r.shuffle(g)

    test, egitim = [], []
    for g in gruplar.values():
        pay = round(args.test * len(g) / len(kurallar))
        test += g[:pay]
        egitim += g[pay:]
    r.shuffle(test)
    r.shuffle(egitim)

    def kayit(k):
        ls = k.get("logsource") or {}
        det = k.get("detection") or {}
        return {
            "baslik": k.get("title", ""),
            "aciklama": (k.get("description") or "").strip(),
            "logsource": {a: ls[a] for a in ("product", "category", "service")
                          if ls.get(a)},
            "detection": {a: v for a, v in det.items() if a != "condition"},
            "condition": str(det.get("condition", "")),
            "attack": kimlikler(k),
            # Kademeli ödülün en alt basamağı: teknik tutmasa da taktik
            # tutuyorsa kısmi puan. Ayrıntı odul.attack_puani().
            "taktik": sorted({str(t).lower().replace("attack.", "")
                              for t in (k.get("tags") or [])
                              if str(t).lower().startswith("attack.")
                              and not T_KIMLIK.match(str(t))}),
            "level": k.get("level", ""),
            "zorluk": zorluk(k),
            "yazar": k.get("author", ""),      # DRL 1.1 atıf şartı
            "kaynak": k["_yol"],
            "ham": k["_ham"],
        }

    def yaz(kayitlar, ad):
        yol = VERI / ad
        with open(yol, "w", encoding="utf-8") as f:
            for k in kayitlar:
                f.write(json.dumps(kayit(k), ensure_ascii=False) + "\n")
        print(f"  {ad:<22} {len(kayitlar):>5} kayıt")

    print("\n=== DAĞILIM ===")
    for eksen, ad in (("zorluk", "condition zorluğu"), ("product", "logsource")):
        say = Counter(
            zorluk(k) if eksen == "zorluk"
            else str((k.get("logsource") or {}).get("product") or "-")
            for k in kurallar)
        print(f"  {ad}:")
        for a, n in say.most_common(5):
            print(f"    {a:<20} {n:>5}")

    print("\n=== YAZILDI ===")
    yaz(egitim, "havuz_egitim.jsonl")
    yaz(test, "havuz_test.jsonl")
    yaz(r.sample(egitim, min(20, len(egitim))), "gozle_oku.jsonl")

    print("\nSıradaki adım: gozle_oku.jsonl dosyasını AÇIP OKUYUN.")


if __name__ == "__main__":
    main()
