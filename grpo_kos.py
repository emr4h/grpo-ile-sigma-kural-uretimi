"""
GRPO eğitimi — Sigma kuralı üretimi.

SIRA BOZULMAZ: SFT → GRPO
    GRPO sıfırdan biçim öğretmez. Eğitimsiz model Sigma kuralı değil düz
    metin üretir, ödül sürekli 0 gelir ve GRPO'nun karşılaştıracak bir şeyi
    olmaz. Bu yüzden `--adapter ckpt/sft_1000` ile SFT modelinden başlıyoruz.

ÖDÜL
    `odul.py` dosyasındaki dört bileşen, üçü kesin. TRL bunları ayrı ayrı
    çağırıp TOPLUYOR. Eğitimden önce `python odul.py --test` ile sınayın —
    yanlış ödülle saatlerce eğitim koşup sonunda modelin puanlayıcıyı
    kandırdığını görmek istemezsiniz.

vLLM — COLOCATE BU İMAJDA ÇALIŞMIYOR
    Plan vLLM'i colocate modunda kullanmaktı; GRPO her adımda 8 cevap
    ürettiği için üretim işin tamamı. Eğitim imajında üç engel çıktı:
    imajda vLLM yok, kurmak torch'u 2.13'e yükseltmek istiyor (Blackwell
    desteği gider), `--no-deps` ile kurulunca ABI uyuşmazlığından import
    çöküyor. Ayrıntı GRPOConfig içindeki notta.

    Ölçülen sonuç: vLLM'siz 14,4 sn/adım. Beklediğimizden iyi, çünkü
    üretilen Sigma kuralları kısa (ortalama 191 token, tavan 768).
    vLLM ayrı konteynerde çalışıyor (vllm_servis.sh, nvcr.io/nvidia/vllm;
    üretim hızı 5,6 kat). GRPO'ya bağlamak için TRL'in vllm_mode="server"
    modu `trl vllm-serve` bekliyor; bu depoda yapılmadı.

KULLANIM
    python odul.py --test                      # ÖNCE bunu koşun
    python grpo_kos.py --adapter ckpt/sft_1000 --adim 150
"""
import argparse
import json
from pathlib import Path

from odul import (A_ATTACK, A_AYRISTIR, A_DONUSTUR, A_ORTUSME, _attack,
                  _taktikler, attack_puani, dogrula)

BURASI = Path(__file__).parent
VERI = BURASI / "veri"

TOHUM = 42
LORA_KATMANLAR = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def _metin(c):
    return c[0]["content"] if isinstance(c, list) else str(c)


# --- ÖDÜL FONKSİYONLARI ----------------------------------------------------
# TRL her birini f(completions, **kwargs) diye çağırıp sonuçları TOPLAR.
# kwargs içinde veri setindeki ek sütunlar geliyor (attack, logsource...).
#
# Üçü tek bir `dogrula()` çağrısına dayanıyor ve TRL onları ayrı ayrı
# çağırdığı için doğrulama üç kez tekrarlanıyor. Ölçüp gerekirse önbellek
# eklenecek; şimdilik netlik hızdan önemli.

def odul_ayristir(completions, **kw):
    """KESİN: pySigma YAML+şemayı kabul ediyor mu?"""
    return [A_AYRISTIR if dogrula(_metin(c))[0] else 0.0 for c in completions]


def odul_donustur(completions, **kw):
    """KESİN: backend condition'ı çözebiliyor mu?

    Ayrı bir bileşen çünkü ölçüldü: from_yaml() tanımsız condition
    referansını kabul ediyor, yalnızca dönüşüm yakalıyor.
    """
    return [A_DONUSTUR if dogrula(_metin(c))[1] else 0.0 for c in completions]


def odul_attack(completions, attack=None, taktik=None, **kw):
    """KESİN ama KADEMELİ: ATT&CK isabeti. Ödülün çapası.

    İkili sürüm GRPO'yu çökertmişti (bkz. odul.attack_puani). Doğru cevap
    390 teknik arasından seyrek çıktığı için sekiz üretimin hepsi 0 alıyor,
    std sıfırlanıyor ve gradyan ölüyordu. Kademeli puan tırmanma imkânı
    veriyor: taktik 0.2, aile 0.5, tam isabet 1.0.
    """
    if attack is None:
        return [0.0] * len(completions)
    n = len(completions)
    tak = taktik or [[]] * n
    out = []
    for c, bek, bek_tak in zip(completions, attack, tak):
        _, _, kural = dogrula(_metin(c))
        if kural is None:
            out.append(0.0)
            continue
        out.append(A_ATTACK * attack_puani(_attack(kural), bek,
                                           _taktikler(kural), bek_tak))
    return out


def odul_ortusme(completions, ls_product=None, ls_category=None,
                 ls_service=None, **kw):
    """KISMİ: logsource tuttu mu? Ağırlığı kasten küçük (0.3).

    Büyütürsek model "referansa benzeyen ama çalışmayan" metin üretmeye
    yönelir — ödülü kandırmanın en kolay yolu budur.

    logsource iç içe dict değil, üç düz sütun olarak geliyor: Arrow şema
    çıkarımı iç içe serbest yapıları kaldıramıyor (bkz. sft_hazirla.py).
    """
    n = len(completions)
    sut = {"product": ls_product or [""] * n,
           "category": ls_category or [""] * n,
           "service": ls_service or [""] * n}
    out = []
    for i, c in enumerate(completions):
        _, _, kural = dogrula(_metin(c))
        if kural is None:
            out.append(0.0)
            continue
        lk = getattr(kural, "logsource", None)
        beklenen = {a: sut[a][i] for a in sut if sut[a][i]}
        if not beklenen:
            out.append(0.0)
            continue
        eslesen = sum(1 for a, v in beklenen.items()
                      if str(getattr(lk, a, "") or "").lower() == str(v).lower())
        out.append(A_ORTUSME * eslesen / len(beklenen))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None,
                    help="SFT adaptörü — GRPO bunun üstüne gelir")
    ap.add_argument("--veri", default=None)
    ap.add_argument("--cikti", default=str(BURASI / "ckpt" / "grpo"))
    ap.add_argument("--adim", type=int, default=150)
    ap.add_argument("--beta", type=float, default=0.02,
                    help="KL cezası. 0 = çıpa yok, biçim sürükleniyor")
    ap.add_argument("--vllm", action="store_true",
                    help="vLLM colocate — eğitim imajında ÇALIŞMIYOR, bkz. GRPOConfig notu")
    args = ap.parse_args()

    if args.veri is None:
        aday = sorted(VERI.glob("grpo_*.jsonl"))
        if not aday:
            print("  ! GRPO verisi yok. Önce: python sft_hazirla.py")
            raise SystemExit(1)
        args.veri = str(aday[0])

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    veri = load_dataset("json", data_files=args.veri, split="train")
    print(f"veri: {len(veri)} prompt  ({Path(args.veri).name})")

    import torch
    if args.adapter:
        # SFT adaptörünü taban ağırlıklara ERİT. GRPO üstüne yeni bir adaptör
        # takacak; iki adaptörü üst üste bindirmek yerine tabanı ilerletmek
        # daha temiz ve vLLM tarafında da sorun çıkarmıyor.
        taban = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(taban, args.adapter).merge_and_unload()
        print(f"  SFT adaptörü eritildi: {args.adapter}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto")

    # --- TRL 0.26.1 <-> transformers 5.12.1 UYUMSUZLUĞU ---------------------
    # GRPOTrainer.__init__ (grpo_trainer.py:501) şunu yapıyor:
    #     model.warnings_issued["estimate_tokens"] = True
    # `warnings_issued`, transformers 4.x'te PreTrainedModel'in bir alanıydı;
    # 5.x'te kaldırılmış. TRL hâlâ arıyor ve AttributeError ile çöküyor.
    #
    # Model string olarak verilseydi de aynı yerde patlardı — TRL modeli
    # kendisi yükleyip aynı satıra geliyor. Bu yüzden nesneyi burada
    # yamalıyoruz. Ölçüldü 16.09.2026.
    #
    # Not: ortam dogrulama scriptim bes trainer'i da "OK" gosteriyordu; yalnizca IMPORT
    # ediyor, KURMAYI denemiyor. İkisi farklı şey.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    ayar = GRPOConfig(
        output_dir=args.cikti,

        # RL'de öğrenme oranı ÇOK küçüktür. Politika büyük adımlarla
        # değişirse model çöker ve geri dönüşü olmaz.
        learning_rate=5e-6,

        # G = prompt başına cevap sayısı. Avantaj grup içinde hesaplandığı
        # için ayrı değer modeli gerekmez; GRPO'nun PPO'dan ucuz olmasının
        # sebebi bu. batch_size buna TAM BÖLÜNMELİ.
        num_generations=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,

        max_prompt_length=512,
        max_completion_length=768,    # Sigma kuralları kısa; 1024 gereksiz

        temperature=1.0,              # 8 cevap birbirinin aynısı olursa
                                      # karşılaştıracak bir şey kalmaz
        # KL CEZASI — kapalı bırakmak biçimi öldürüyor (ölçüldü 16.09.2026).
        # beta=0.0 ile modeli SFT başlangıç noktasına bağlayan hiçbir şey
        # yok; RL serbestçe sürükleniyor. İki koşuda da aynı desen çıktı:
        #   ayristir  0.419 -> 0.369      donustur  0.812 -> 0.738
        # Kademeli ödül `attack`i tırmandırdı (0.048 -> 0.104) ama biçim
        # kaybı kazancı aşıyordu: +0.056 kazanç, -0.075 kayıp.
        # beta modeli referans politikaya çıpalıyor.
        beta=args.beta,

        max_steps=args.adim,
        bf16=True,

        # --- vLLM COLOCATE BU İMAJDA KULLANILAMIYOR (ölçüldü 16.09.2026) ---
        # Plan colocate modunda vLLM kullanmaktı; GRPO her adımda 8 cevap
        # ürettiği için üretim hızı işin tamamı. Üç engel çıktı:
        #
        # 1. NVIDIA playbook imajı vLLM kurmuyor. GRPOTrainer use_vllm=True
        #    ile kurulurken ImportError veriyor (grpo_trainer.py:613).
        #    weave/mergekit eksikliğiyle aynı sınıf bir boşluk.
        # 2. `pip install vllm` torch'u 2.13.0'a yükseltmek istiyor. NGC'nin
        #    2.10.0a0 derlemesi Blackwell (sm_121) desteğini taşıyor;
        #    değiştirmek GPU'yu kullanılamaz hale getirir.
        # 3. `--no-deps` ile kurulunca import çöküyor:
        #      vllm/_C_stable_libtorch.abi3.so:
        #      undefined symbol: torch_get_mutable_data_ptr
        #    Derlenmiş uzantı torch'un C++ ABI'sine bağlı ve 2.10 ile 2.13
        #    uyumlu değil.
        #
        # torchao hikâyesiyle aynı ders: çözüm yükseltmek değil, vazgeçmek.
        # Üretim `transformers.generate` ile yapılıyor — yavaş ama çalışıyor.
        # Sonradan: "bu cihazda olmaz" genellemesi yanlıştı. vLLM AYRI
        # konteynerde (nvcr.io/nvidia/vllm) çalışıyor, bkz. vllm_servis.sh.
        # GRPO'ya bağlamak vllm_mode="server" + `trl vllm-serve` ister;
        # sıradaki iş.
        use_vllm=args.vllm,
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": 0.3}
           if args.vllm else {}),

        logging_steps=5,
        seed=TOHUM,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[odul_ayristir, odul_donustur, odul_attack, odul_ortusme],
        args=ayar,
        train_dataset=veri,
        peft_config=LoraConfig(r=32, lora_alpha=64,   # SFT'den büyük rank
                               target_modules=LORA_KATMANLAR,
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model()
    print(f"\nBitti -> {args.cikti}")

    # --- NASIL OKUNUR ------------------------------------------------------
    # reward          : ortalama ödül. YÜKSELMELİ. Tavan 2.8.
    # kl              : küçük kalmalı; fırlıyorsa beta ekleyin
    #
    # Ödül hızla tavana yapışıyorsa -> görev kolay, promptları zorlaştırın
    # Ödül hiç kalkmıyorsa          -> SFT yetersiz VEYA ödül bozuk
    #                                  (python odul.py --test ile bakın)


if __name__ == "__main__":
    main()
