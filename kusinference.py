import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.models as models
import librosa
import numpy as np
import gradio as gr
import os

MODEL_PATH = "resnet_kus_5li.pth"

DATA_PATH = "mp3_sliced"
LABELS_PATH = TARGET_BIRDS = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])

# Kontrol için yazdıralım
print(f" Tespit Edilen Sınıf Sayısı: {len(TARGET_BIRDS)}")
print(f" Örnek Sınıflar: {TARGET_BIRDS[:5]} ...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 22050
DURATION = 4.0  # Saniye
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 22050 * 4

try:
    # Klasörden otomatik okuma (Alfabetik sıra çok önemli!)
    TARGET_BIRDS = sorted([
        d for d in os.listdir(LABELS_PATH)
        if os.path.isdir(os.path.join(LABELS_PATH, d))
    ])
    print(f" {len(TARGET_BIRDS)} adet sınıf başarıyla yüklendi.")
except Exception as e:
    print(" UYARI: Klasör yolu bulunamadı. Lütfen TARGET_BIRDS listesini elle doldurun.")
    # Örnek: TARGET_BIRDS = ['Kus1', 'Kus2', ...]
    TARGET_BIRDS = []

class ResNetBirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBirdClassifier, self).__init__()
        # Inference yapacağımız için weights=None diyebiliriz (dosyadan yükleyeceğiz)
        self.model = models.resnet18(weights=None)

        # Giriş Katmanı (1 Kanal - Spektrogram)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Çıkış Katmanı (Dropout + Linear)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def predict_bird_smart(audio_path):
    if not audio_path:
        return "Lütfen bir ses dosyası yükleyin."

    # Modeli Yükle
    model = ResNetBirdClassifier(len(TARGET_BIRDS)).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        return f" Hata: Model yüklenemedi. Yol doğru mu?\n{e}"

    # Sesi Yükle
    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        return f"Ses dosyası okunamadı: {e}"

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=1024,
        n_mels=128,
        f_min=500,
        f_max=10000
    ).to(DEVICE)

    db_transform = T.AmplitudeToDB(top_db=80).to(DEVICE)

    # Ses dosyasını 4 saniyelik parçalara ayırıp hepsini modele soracağız.
    batch_tensors = []

    # Adım adım ilerle (hop olmadan, direkt 4sn, 4sn diye bölelim)
    for i in range(0, len(signal), NUM_SAMPLES):
        chunk = signal[i: i + NUM_SAMPLES]

        # parça çok kısaysa dikkate alma
        if len(chunk) < SAMPLE_RATE:
            continue

        # Kısa parçayı doldur (Padding)
        if len(chunk) < NUM_SAMPLES:
            chunk = np.pad(chunk, (0, NUM_SAMPLES - len(chunk)))

        # Tensöre çevir
        chunk_tensor = torch.from_numpy(chunk).float().to(DEVICE)
        chunk_tensor = chunk_tensor.unsqueeze(0)  # Channel ekle (1, Time)

        # Spektrograma çevir
        spec = mel_transform(chunk_tensor)
        spec = db_transform(spec)
        batch_tensors.append(spec)

    if not batch_tensors:
        return "Ses dosyası çok kısa veya sessiz."

    batch_stack = torch.stack(batch_tensors).to(DEVICE)  # (N, 1, 128, Time)

    with torch.no_grad():
        outputs = model(batch_stack)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        avg_probs = torch.mean(probs, dim=0)

    results = {
        TARGET_BIRDS[i]: float(avg_probs[i])
        for i in range(len(TARGET_BIRDS))
    }

    return results

interface = gr.Interface(
    fn=predict_bird_smart,
    inputs=gr.Audio(type="filepath", label="Kuş Sesi Yükle veya Kaydet"),
    outputs=gr.Label(num_top_classes=5, label="Tahmin Sonuçları"),
    title="🦅 AvianAcoustics: Profesyonel Kuş Tanıma",
    description=f"Bu sistem **{len(TARGET_BIRDS)} farklı kuş türünü** tanıyabilir. Uzun bir kayıt yüklediğinizde, yapay zeka kaydın tamamını dinleyip genel bir karar verir.",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    print("🚀 Arayüz başlatılıyor...")
    # share=True diyerek herkese açık link oluşturabilirsin
    interface.launch(share=True)