"""
LoRA + SFT eğitimi — bu deneylerde kullanılan eğitim scripti.

NEDİR?
    SFT  = "soru -> ideal cevap" çiftleriyle modele davranış öğretmek.
    LoRA = bunu yaparken modelin tamamını değil, yanına eklenen küçük
           matrisleri eğitmek. Bellek 3-5 kat düşer, sonuç neredeyse aynı.

BU DEPODAKİ YERİ
    `deney2_veri.py --kur` üç eğitim seti üretir (deney2_k0/k8/k40.jsonl).
    Bu script her birini ayrı ayrı eğitir; `deney2_test.py` sonucu ölçer.

GİRDİ FORMATI
    Her satır bir JSON nesnesi:
      {"messages": [{"role":"system",...},{"role":"user",...},
                    {"role":"assistant",...}]}
    Chat şablonunu tokenizer otomatik uygular.

KULLANIM
    python egit_lora.py --veri veri/deney2_k40.jsonl --cikti ckpt/k40 --epoch 3
    python egit_lora.py --model Qwen/Qwen3-8B --veri veri/deney2_k40.jsonl \
                        --cikti ckpt/8b_k40 --epoch 3

EĞİTİM NASIL OKUNUR?
    train_loss : düzgün düşmeli. Tipik 1,5-2,5 -> 0,5-1,0
    eval_loss  : o da düşmeli. YÜKSELMEYE BAŞLARSA DURUN — ezberliyor.
    0,1'in altı: büyük olasılıkla ezber, daha az epoch deneyin.
    Kayıp fırlıyorsa (nan / 10+): learning_rate'i yarıya indirin.
"""
import argparse
import sys

TOHUM = 42
MAX_UZUNLUK = 2048

# LoRA ayarları
LORA_R = 16          # rank: eklenen matrislerin "kalınlığı"
                     #   8-16  -> üslup/format öğretmek için yeter
                     #   32-64 -> yeni görev veya alan bilgisi
LORA_ALPHA = 32      # ölçek katsayısı. Yaygın kural: alpha = 2 x r
LORA_DROPOUT = 0.05  # küçük veride ezberi frenler

# Tüm doğrusal katmanlar. Yalnızca dikkat katmanları (ilk 4) daha hızlıdır
# ama daha az öğrenir; yedisi birden belirgin daha iyi sonuç veriyor.
LORA_KATMANLAR = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # dikkat mekanizması
    "gate_proj", "up_proj", "down_proj",         # ileri beslemeli ağ
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--veri", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--epoch", type=float, default=3)
    ap.add_argument("--r", type=int, default=LORA_R)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Çok kipli (vision-language) modellerde TRL assistant_only_loss'u henüz
    # desteklemiyor. Bu deneylerde saf metin modelleri kullanıldı.
    cfg = AutoConfig.from_pretrained(args.model)
    if hasattr(cfg, "vision_config"):
        print(f"  ! {args.model} çok kipli bir model; saf metin modeli kullanın "
              f"(örn. Qwen/Qwen3-0.6B)")
        sys.exit(1)

    veri = load_dataset("json", data_files=args.veri, split="train")

    # ASİSTAN MASKESİ — kayıp yalnızca asistan cevabında hesaplanmalı.
    # Kapatırsanız model kullanıcının sorularını da taklit etmeyi öğrenir ve
    # kendi kendine soru sormaya başlar.
    #
    # assistant_only_loss=True, sohbet şablonunda {% generation %} etiketi
    # arar. Qwen şablonunda bu etiket YOKTUR -> RuntimeError. Çözüm veriyi
    # prompt/completion formatına çevirmek: TRL o formatta prompt'u zaten
    # otomatik maskeliyor, sonuç aynı.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sablon = tokenizer.chat_template or ""
    maske_destegi = "{% generation %}" in sablon or "{%- generation" in sablon
    sadece_asistan = True

    if not maske_destegi and "messages" in veri.column_names:
        print("  ! Şablonda {% generation %} yok -> prompt/completion formatına çevriliyor")
        print("    (sonuç aynı: kayıp yalnızca asistan cevabında hesaplanır)")

        def bol(ornek):
            m = ornek["messages"]
            son = len(m) - 1
            while son > 0 and m[son]["role"] != "assistant":
                son -= 1
            return {"prompt": m[:son], "completion": [m[son]]}

        veri = veri.map(bol, remove_columns=["messages"])
        sadece_asistan = False      # format hallediyor

    # Doğrulama seti zorunlu değil ama şiddetle önerilir: aşırı öğrenmenin
    # tek erken uyarı sinyali eval_loss'tur.
    veri = veri.train_test_split(test_size=0.1, seed=TOHUM)
    print(f"veri: {len(veri['train'])} eğitim / {len(veri['test'])} doğrulama")

    # Orijinal ağırlık W dondurulur; yanına iki ince matris eklenir:
    #   A (giris x r) ve B (r x cikis).  Çıktı: W·x + (alpha/r)·B·A·x
    lora = LoraConfig(
        r=args.r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_KATMANLAR,
        task_type="CAUSAL_LM",
    )

    ayar = SFTConfig(
        output_dir=args.cikti,
        num_train_epochs=args.epoch,

        # Etkin batch = 4 x 4 = 16. Bellek yetmezse batch'i yarıya indirip
        # accumulation'ı iki katına çıkarın: etkin batch aynı kalır.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        # En kritik ayar. LoRA'da 1e-4 ~ 2e-4 tipiktir — tam ince ayardan 10
        # kat büyük, çünkü sıfırdan başlayan küçük matrisleri eğitiyoruz.
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        max_length=MAX_UZUNLUK,
        bf16=True,
        assistant_only_loss=sadece_asistan,
        packing=False,

        eval_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        seed=TOHUM,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model,
        args=ayar,
        train_dataset=veri["train"],
        eval_dataset=veri["test"],
        peft_config=lora,      # bu satır olmasaydı TAM ince ayar olurdu
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model()       # yalnızca adaptör kaydedilir
    print(f"\nBitti -> {args.cikti}")


if __name__ == "__main__":
    main()
