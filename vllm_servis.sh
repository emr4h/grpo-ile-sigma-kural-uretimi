#!/usr/bin/env bash
# vLLM'i AYRI KONTEYNERDE servis eder.
#
# NEDEN AYRI KONTEYNER?
#   `pip install vllm` eğitim imajına kurulamıyor. Üç katmanlı sorun:
#     1. NVIDIA playbook imajı vLLM kurmuyor
#     2. pip ile kurmak torch'u 2.13'e yükseltmek istiyor — NGC'nin
#        2.10.0a0 derlemesi Blackwell (sm_121) desteğini taşıyor, gitmesi
#        GPU'yu kullanılamaz yapar
#     3. `--no-deps` ile kurulunca import çöküyor:
#          undefined symbol: torch_get_mutable_data_ptr
#        Derlenmiş uzantı torch'un C++ ABI'sine bağlı; 2.10 ile 2.13 uyumsuz
#
#   İLK SONUCUM YANLIŞTI. "Kurulmuyorsa bu cihazda olmaz" diye genelledim.
#   Oysa NVIDIA kendi torch'una karşı derlenmiş bir vLLM imajı yayınlıyor:
#   nvcr.io/nvidia/vllm, arm64 dahil. Tek kurulum yolu denemek yetmiyormuş.
#
#   Doğru mimari iki konteyner: vLLM kendi torch'uyla servis eder, eğitim
#   konteyneri HTTP üzerinden konuşur. ABI çakışması ortadan kalkar.
#
# KULLANIM
#   ./vllm_servis.sh baslat ckpt/sft_merged   # servisi ayağa kaldır
#   ./vllm_servis.sh durum                    # hazır mı?
#   ./vllm_servis.sh durdur
#
#   Sonra eğitim tarafında:
#     python grpo_kos.py --vllm-server --adapter ckpt/sft_1000 --adim 200

set -uo pipefail
BURASI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAJ="${VLLM_IMAJ:-nvcr.io/nvidia/vllm:25.11-py3}"
AD="vllm-sigma"
PORT="${VLLM_PORT:-8000}"

baslat() {
    local model="${1:-Qwen/Qwen3-0.6B}"
    if docker ps --filter "name=$AD" --format '{{.ID}}' | grep -q .; then
        echo "  zaten çalışıyor: $AD"; return 0
    fi
    echo "  vLLM başlatılıyor: $model  (port $PORT)"
    # --net=host: TRL aynı makineden localhost:8000'e bağlanacak
    # --gpus=all: GPU paylaşımlı; eğitim de aynı kartı kullanacağı için
    #   gpu-memory-utilization düşük tutuluyor
    docker run -d --rm --name "$AD" --gpus=all --ipc=host --net=host \
        -v "$HOME/.cache/huggingface":/hf -e HF_HOME=/hf \
        -v "$BURASI":/p \
        "$IMAJ" \
        vllm serve "$model" \
            --port "$PORT" \
            --gpu-memory-utilization 0.35 \
            --max-model-len 2048 \
            --dtype bfloat16 >/dev/null
    echo "  konteyner: $(docker ps --filter "name=$AD" --format '{{.ID}}')"
    echo "  hazır olmasını beklemek için: ./vllm_servis.sh durum"
}

durum() {
    if ! docker ps --filter "name=$AD" --format '{{.ID}}' | grep -q .; then
        echo "  çalışmıyor"; return 1
    fi
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "  HAZIR — http://localhost:$PORT"
        curl -s "http://localhost:$PORT/v1/models" | head -c 200; echo
    else
        echo "  ayakta ama henüz yüklüyor (model indiriliyor/ısınıyor)"
        docker logs --tail 5 "$AD" 2>&1 | tail -3
    fi
}

durdur() {
    docker stop "$AD" >/dev/null 2>&1 && echo "  durduruldu" || echo "  zaten kapalı"
}

case "${1:-}" in
    baslat) shift; baslat "${1:-Qwen/Qwen3-0.6B}" ;;
    durum)  durum ;;
    durdur) durdur ;;
    *) sed -n '2,30p' "$0" ;;
esac
