"""
Havuzu SFT eğitim formatına çevirir.

NEDEN ÖNCE SFT?
    GRPO sıfırdan biçim öğretmez; var olan davranışı iyileştirir. Eğitimsiz
    bir model Sigma kuralı değil düz metin üretir, ödül sürekli 0 gelir ve
    GRPO'nun karşılaştıracak bir şeyi olmaz. Önce SFT ile YAML iskeletini
    oturtuyoruz, sonra GRPO doğru alanları ve doğru ATT&CK etiketini
    bulmayı öğretiyor.

    Sıra bozulmaz: SFT → GRPO.

BÖLÜNME
    Havuz ikiye ayrılır ve KESİŞMEZ:
      sft_*.jsonl   → SFT eğitimi (soru + ideal cevap çiftleri)
      grpo_*.jsonl  → GRPO promptları (yalnızca soru; cevabı model üretecek)

    Aynı kayıt ikisinde birden olursa GRPO, SFT'nin ezberlediği cevabı
    "keşfetmiş" gibi görünür ve kazanç sahte çıkar.

KULLANIM
    python sft_hazirla.py                    # 1000 SFT + kalanı GRPO
    python sft_hazirla.py --sft 500
"""
import argparse
import json
import random
from pathlib import Path

BURASI = Path(__file__).parent
VERI = BURASI / "veri"

# Kısa tutuluyor: modelin her seferinde uzun kuralları okumasına gerek yok,
# davranış ağırlıklara işleyecek. Uzun sistem promptu hem token yakar hem
# modelin talimatı ezberlemesine yol açar.
SISTEM = ("You are a detection engineer. Given a description of suspicious "
          "behaviour, you write a valid Sigma detection rule in YAML, "
          "including the matching MITRE ATT&CK technique tag.")


def soru(k):
    """Modele verilecek istem. Logsource ipucu olarak veriliyor —
    hangi log kaynağına kural yazılacağı görevin girdisi, çıkarımı değil."""
    ls = k.get("logsource") or {}
    satir = ", ".join(f"{a}: {v}" for a, v in ls.items())
    return (f"Write a Sigma rule for the following detection:\n\n"
            f"{k['aciklama']}\n\n"
            f"Log source — {satir}")


# Modelin ÖĞRENEMEYECEĞİ alanlar — ideal cevaptan çıkarılıyor.
#
# Ölçüldü (16.09.2026): ham kuralların hepsinde bu alanlar var ve hiçbiri
# açıklamadan türetilemiyor. Bırakırsak üç ayrı zarar veriyorlar:
#
#   1. Eğitim sinyalinin yarısını öğrenilemez metadata'ya harcarız.
#      Cevaplar %40 uzuyor, kazanç sıfır.
#   2. Modeli halüsinasyona alıştırırız: uydurma UUID, var olmayan blog
#      linkleri, sahte tarihler.
#   3. `author` en kötüsü — DRL 1.1 atıf şartını İHLAL eder. Model uydurma
#      yazar adı yazarsa kuralı yazmayan kişilere atfetmiş oluruz.
#      Yazar bilgisi veri setinin metadata'sında (`yazar` alanı) ve
#      README'de duruyor; modelin çıktısında değil.
#
# Doğrulama etkilenmiyor: sadeleştirilmiş kural da 2.80 alıyor (ölçüldü).
ATILACAK = {"id", "author", "date", "modified", "references", "related",
            "regression_tests_path", "simulation", "status"}


def cevap(k):
    """İdeal cevap: kuralın öğrenilebilir alanları, kod çitiyle sarılmış."""
    import yaml
    d = yaml.safe_load(k["ham"])
    sade = {a: v for a, v in d.items() if a not in ATILACAK}
    metin = yaml.dump(sade, allow_unicode=True, sort_keys=False,
                      default_flow_style=False).strip()
    return f"```yaml\n{metin}\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default=str(VERI / "havuz_egitim.jsonl"))
    ap.add_argument("--sft", type=int, default=1000)
    ap.add_argument("--tohum", type=int, default=42)
    args = ap.parse_args()

    kayitlar = [json.loads(s) for s in
                open(args.girdi, encoding="utf-8") if s.strip()]
    print(f"havuz: {len(kayitlar)} kayıt")

    # Zorluk dağılımını iki tarafta da koru. SFT yalnızca kolay örnekler
    # görürse GRPO'ya zayıf bir başlangıç noktası bırakır.
    r = random.Random(args.tohum)
    gruplar = {}
    for k in kayitlar:
        gruplar.setdefault(k["zorluk"], []).append(k)
    for g in gruplar.values():
        r.shuffle(g)

    sft, grpo = [], []
    for tur, g in sorted(gruplar.items()):
        pay = round(args.sft * len(g) / len(kayitlar))
        sft += g[:pay]
        grpo += g[pay:]
        print(f"  {tur:<12} {len(g):>5} → SFT {pay}, GRPO {len(g)-pay}")
    r.shuffle(sft)
    r.shuffle(grpo)

    yol = VERI / f"sft_{len(sft)}.jsonl"
    with open(yol, "w", encoding="utf-8") as f:
        for k in sft:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SISTEM},
                {"role": "user", "content": soru(k)},
                {"role": "assistant", "content": cevap(k)},
            ]}, ensure_ascii=False) + "\n")
    print(f"\n  {yol.name:<22} {len(sft):>5} kayıt")

    # GRPO tarafında cevap YOK — model üretecek. Ama ödül için referans
    # bilgisi lazım: beklenen ATT&CK kimlikleri ve logsource.
    #
    # ŞEMA DÜZ TUTULUYOR. `datasets` kütüphanesi Arrow kullanıyor ve tüm
    # satırların aynı tipte olmasını istiyor. Sigma'nın `detection` alanı
    # serbest biçimli: bir kuralda `EventID: 4104` (sayı), başkasında
    # `EventID: [4104, 4105]` (dizi). Arrow tek tip belirleyemeyip patlıyor:
    #   ArrowInvalid: Column(/detection/selection/EventID)
    #                 changed from number to array in row 7
    # Ölçüldü 16.09.2026. Zaten `detection`'ı ödül kullanmıyor; logsource'u
    # da iç içe dict yerine düz üç sütuna açıyoruz.
    ls_anahtar = ("product", "category", "service")
    yol = VERI / f"grpo_{len(grpo)}.jsonl"
    with open(yol, "w", encoding="utf-8") as f:
        for k in grpo:
            ls = k.get("logsource") or {}
            f.write(json.dumps({
                "prompt": [{"role": "system", "content": SISTEM},
                           {"role": "user", "content": soru(k)}],
                "attack": k["attack"],                    # list[str] — Arrow uyumlu
                "taktik": k.get("taktik") or [],           # kademeli ödülün alt basamağı
                **{f"ls_{a}": str(ls.get(a) or "") for a in ls_anahtar},
                "zorluk": k["zorluk"],
            }, ensure_ascii=False) + "\n")
    print(f"  {yol.name:<22} {len(grpo):>5} kayıt")
    print("\n  (kesişim yok: bir kayıt ya SFT'de ya GRPO'da)")


if __name__ == "__main__":
    main()
