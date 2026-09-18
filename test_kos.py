"""
Modeli test setinde koşturur ve ödül fonksiyonuyla puanlar.

NEDEN AYRI ADIM?
    GRPO'nun kazancını ölçebilmek için ÖNCE taban puan lazım. "GRPO +11
    puan kazandırdı" demek, ancak öncesini ölçtüyseniz anlamlı. Bu script
    hem eğitimsiz modeli hem SFT modelini hem de GRPO modelini aynı ölçekle
    puanlıyor.

ZORLUĞA GÖRE BÖLÜNÜYOR
    Tek ortalama yanıltır. Model kolay kurallarda (tek seçim) iyi görünüp
    zor olanlarda (mantıksal condition) çuvallayabilir; toplam skor bunu
    gizler. Sonuç zorluk kırılımıyla basılıyor.

KULLANIM
    python test_kos.py --cikti cikti/temel.jsonl                    # eğitimsiz
    python test_kos.py --adapter ckpt/sft_1000 --cikti cikti/sft.jsonl
    python test_kos.py --adapter ckpt/grpo    --cikti cikti/grpo.jsonl
    python test_kos.py --olc cikti/sft.jsonl                        # yeniden puanla
"""
import argparse
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

from odul import puanla
from sft_hazirla import SISTEM, soru

BURASI = Path(__file__).parent
VERI = BURASI / "veri"


def raporla(kayitlar, ad=""):
    if not kayitlar:
        print("  (kayıt yok)")
        return
    bilesen = defaultdict(list)
    zorluk = defaultdict(list)
    for k in kayitlar:
        p = k["puan"]
        for a, v in p.items():
            bilesen[a].append(v)
        bilesen["TOPLAM"].append(sum(p.values()))
        zorluk[k.get("zorluk", "?")].append(sum(p.values()))

    print(f"\n=== {ad} · {len(kayitlar)} kayıt ===")
    for a in ("ayristir", "donustur", "attack", "ortusme", "TOPLAM"):
        if a in bilesen:
            print(f"  {a:<10}{st.mean(bilesen[a]):>8.3f}")
    print("  zorluğa göre:")
    for z in ("tek", "toplu", "mantiksal"):
        if z in zorluk:
            print(f"    {z:<12}{st.mean(zorluk[z]):>8.3f}  (n={len(zorluk[z])})")
    tam = sum(1 for k in kayitlar if k["puan"]["donustur"] > 0)
    print(f"  çalışan kural üretme oranı: {tam}/{len(kayitlar)} "
          f"(%{100*tam/len(kayitlar):.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--taban-adapter", dest="taban_adapter", default=None,
                    help="Önce eritilecek adaptör (GRPO için SFT adaptörü)")
    ap.add_argument("--test", default=str(VERI / "havuz_test.jsonl"))
    ap.add_argument("--cikti", default=None)
    ap.add_argument("--adet", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-yeni", type=int, default=700)
    ap.add_argument("--olc", default=None, help="Var olan çıktıyı yeniden puanla")
    args = ap.parse_args()

    if args.olc:
        k = [json.loads(s) for s in open(args.olc, encoding="utf-8") if s.strip()]
        raporla(k, Path(args.olc).name)
        return
    if not args.cikti:
        ap.error("--cikti gerekli")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kayitlar = [json.loads(s) for s in
                open(args.test, encoding="utf-8") if s.strip()][:args.adet]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")

    # --- ADAPTÖR ZİNCİRİ -------------------------------------------------
    # GRPO, SFT adaptörü ERİTİLMİŞ model üstünde eğitiliyor. Yani ckpt/grpo
    # adaptörü "taban + SFT" ağırlıklarına göreli. Ama PEFT bunu bilmiyor:
    # adapter_config.json'a `base_model: Qwen/Qwen3-0.6B` yazıyor, çünkü
    # from_pretrained'e verilen ad buydu. KONFİGÜRASYON YALAN SÖYLÜYOR.
    #
    # Çıplak tabana takınca model Sigma değil, "tespit kuralı herhalde böyle
    # bir şeydir" tahmini üretiyor (`name:`, `mitre_attack_technique:` gibi
    # uydurma alanlar) ve puan 0.000 çıkıyor. Ölçüldü 16.09.2026; eğitim
    # ödülü 1.6 iken test 0.000 gelince yakalandı.
    #
    # Doğru yol: eğitimdeki tabanı AYNEN yeniden kur.
    from peft import PeftModel
    if args.taban_adapter:
        model = PeftModel.from_pretrained(model, args.taban_adapter).merge_and_unload()
        print(f"  taban adaptörü eritildi: {args.taban_adapter}", flush=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"model: {args.model}"
          + (f" + {args.taban_adapter}(eritilmiş)" if args.taban_adapter else "")
          + (f" + {args.adapter}" if args.adapter else " (eğitimsiz)"), flush=True)

    def istem(k):
        m = [{"role": "system", "content": SISTEM},
             {"role": "user", "content": soru(k)}]
        try:
            return tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True)

    istemler = [istem(k) for k in kayitlar]
    sonuc, t0 = [], time.time()
    for i in range(0, len(istemler), args.batch):
        g = tok(istemler[i:i + args.batch], return_tensors="pt", padding=True,
                truncation=True, max_length=1200).to(model.device)
        with torch.no_grad():
            c = model.generate(**g, max_new_tokens=args.max_yeni,
                               temperature=0.3, do_sample=True, top_p=0.9,
                               pad_token_id=tok.pad_token_id)
        for j in range(len(c)):
            sonuc.append(tok.decode(c[j][g["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip())
        print(f"  {len(sonuc)}/{len(istemler)}  {time.time()-t0:.0f} sn", flush=True)

    cikti = []
    for k, metin in zip(kayitlar, sonuc):
        p, not_ = puanla(metin, k)
        cikti.append({"kaynak": k["kaynak"], "zorluk": k["zorluk"],
                      "attack_beklenen": k["attack"], "uretilen": metin,
                      "puan": p, "not": not_})

    Path(args.cikti).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cikti, "w", encoding="utf-8") as f:
        for r in cikti:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    raporla(cikti, Path(args.cikti).name)
    print(f"\n-> {args.cikti}  ({time.time()-t0:.0f} sn)")


if __name__ == "__main__":
    main()
