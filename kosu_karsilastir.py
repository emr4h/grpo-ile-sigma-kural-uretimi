"""
GRPO koşularını karşılaştırır — eğim çıkarır, tek noktalara bakmaz.

NEDEN REGRESYON?
    Ödül eğrisi gürültülü. Bu pakette bir koşunun ardışık ölçümleri
    1.629 -> 1.364 -> 1.527 gibi zıplıyor. Dört noktaya bakıp "yükseliyor"
    demek yanlış sonuç veriyor — bir kez yaptım, düzelttim.

    Bunun yerine bütün ölçümlere doğru uyduruyoruz. Eğim pozitifse
    öğreniyor, negatifse eroze oluyor. Karar tek sayıya iniyor.

KULLANIM
    python kosu_karsilastir.py /tmp/grpo_train.log /tmp/grpo2.log /tmp/grpo3.log
    python kosu_karsilastir.py --ad "ikili" a.log --ad "kademeli" b.log
"""
import argparse
import re
from pathlib import Path

KALIP = re.compile(
    r"\{[^{}]*'rewards/odul_ayristir/mean': '([\d.]+)'[^{}]*"
    r"'rewards/odul_donustur/mean': '([\d.]+)'[^{}]*"
    r"'rewards/odul_attack/mean': '([\d.]+)'[^{}]*"
    r"'rewards/odul_ortusme/mean': '([\d.]+)'[^{}]*"
    r"'reward': '([\d.-]+)'[^{}]*\}")

BILESEN = ("ayristir", "donustur", "attack", "ortusme", "TOPLAM")


def egim(y):
    """En küçük kareler eğimi. x = ölçüm sırası."""
    n = len(y)
    if n < 2:
        return 0.0
    x = list(range(n))
    ox, oy = sum(x) / n, sum(y) / n
    ust = sum((a - ox) * (b - oy) for a, b in zip(x, y))
    alt = sum((a - ox) ** 2 for a in x)
    return ust / alt if alt else 0.0


def oku(yol):
    t = Path(yol).read_text(encoding="utf-8", errors="replace")
    seri = {a: [] for a in BILESEN}
    for m in KALIP.finditer(t):
        a, d, k, o, top = (float(x) for x in m.groups())
        for ad, v in zip(BILESEN, (a, d, k, o, top)):
            seri[ad].append(v)
    return seri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loglar", nargs="+")
    ap.add_argument("--ad", action="append", default=None)
    args = ap.parse_args()

    adlar = args.ad or [Path(y).stem for y in args.loglar]
    kosular = [(ad, oku(y)) for ad, y in zip(adlar, args.loglar)]

    print("=== ÖLÇÜM SAYISI ===")
    for ad, s in kosular:
        print(f"  {ad:<22} {len(s['TOPLAM'])}")

    for baslik, alan in (("İLK ÖLÇÜM", 0), ("SON ÖLÇÜM", -1)):
        print(f"\n=== {baslik} ===")
        print(f"  {'koşu':<22}" + "".join(f"{a:>10}" for a in BILESEN))
        for ad, s in kosular:
            if not s["TOPLAM"]:
                continue
            print(f"  {ad:<22}" + "".join(f"{s[a][alan]:>10.3f}" for a in BILESEN))

    print("\n=== EĞİM (ölçüm başına değişim; + öğreniyor, − eroze) ===")
    print(f"  {'koşu':<22}" + "".join(f"{a:>10}" for a in BILESEN))
    for ad, s in kosular:
        if len(s["TOPLAM"]) < 2:
            continue
        print(f"  {ad:<22}" + "".join(f"{egim(s[a]):>+10.4f}" for a in BILESEN))

    print("\n  NASIL OKUNUR")
    print("  TOPLAM eğimi negatifse koşu işe yaramıyor — model kaybediyor.")
    print("  attack eğimi pozitif ama TOPLAM negatifse: hedef bileşen")
    print("  öğreniliyor ama biçim eroze oluyor, KL çıpası gerekiyor.")


if __name__ == "__main__":
    main()
