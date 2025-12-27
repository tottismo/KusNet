import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import optuna
import torch.optim as optim
import torch.nn as nn

import torchvision.models as models

DATA_PATH = r"C:\Users\hp\PycharmProjects\pythonProject1\mp3_sliced"
MODEL_SAVE_PATH = "resnet_kus_5li.pth"

# Hedef Kuşlar
TARGET_BIRDS = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])

# Kontrol için yazdıralım
print(f"✅ Tespit Edilen Sınıf Sayısı: {len(TARGET_BIRDS)}")
print(f"📋 Örnek Sınıflar: {TARGET_BIRDS[:5]} ...")

# Hiperparametreler
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 4
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(" Çalışma Ortamı: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU Modeli: {torch.cuda.get_device_name(0)}")

def add_noise(signal, noise_factor=0.005):
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_factor * noise
    return augmented_signal.astype(np.float32)


def plot_confusion_matrix(model, loader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karmaşıklık Matrisi')
    plt.show()

def create_dataframe(data_path, target_classes):
    data = []

    found_counts = {k: 0 for k in target_classes}

    for class_name in os.listdir(data_path):
        if class_name not in target_classes:
            continue

        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.mp3', '.wav')):
                file_path = os.path.join(class_dir, filename)
                data.append([file_path, class_name])
                found_counts[class_name] += 1

    df = pd.DataFrame(data, columns=['file_path', 'label'])
    class_map = {label: idx for idx, label in enumerate(target_classes)}

    print(f" Toplam Dosya: {len(df)}")
    print(f"  Dağılım: {found_counts}")
    return df, class_map

class BirdSoundDataset(Dataset):
    def __init__(self, dataframe, class_mapping, transformation, target_sr, num_samples, device, is_train=False):
        self.annotations = dataframe
        self.class_mapping = class_mapping
        self.transformation = transformation.to(device)
        self.target_sr = target_sr
        self.num_samples = num_samples
        self.device = device
        self.is_train = is_train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self.annotations.iloc[index, 0]
        label_str = self.annotations.iloc[index, 1]
        label = self.class_mapping[label_str]

        signal, sr = librosa.load(audio_path, sr=self.target_sr)

        if self.is_train and np.random.rand() > 0.5:
            signal = add_noise(signal)

        signal = torch.from_numpy(signal).float()
        signal = signal.unsqueeze(0)
        signal = signal.to(self.device)

        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            pad_amount = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, pad_amount))

        signal = self.transformation(signal)

        return signal, label

class ResNetBirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBirdClassifier, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_features = self.model.fc.in_features  #özellik sayısı (512)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def main():
    df, class_mapping = create_dataframe(DATA_PATH, TARGET_BIRDS)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=1024, n_mels=128, f_min=500, f_max=10000
    )

    train_transform = nn.Sequential(
        mel_spectrogram,
        T.AmplitudeToDB(top_db=80),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=60)
    )

    test_transform = nn.Sequential(
        mel_spectrogram,
        T.AmplitudeToDB(top_db=80)
    )

    train_dataset = BirdSoundDataset(train_df, class_mapping, train_transform, SAMPLE_RATE, NUM_SAMPLES, DEVICE,
                                     is_train=True)
    test_dataset = BirdSoundDataset(test_df, class_mapping, test_transform, SAMPLE_RATE, NUM_SAMPLES, DEVICE,
                                    is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNetBirdClassifier(len(TARGET_BIRDS)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train_acc': [], 'test_acc': []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct = 0;
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)

        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        scheduler.step()

        model.eval()
        test_correct = 0;
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_msg = "REKOR!"
        else:
            save_msg = ""

        print(
            f"   LR: {scheduler.get_last_lr()[0]:.6f} | Train Acc: %{train_acc:.2f} | Test Acc: %{test_acc:.2f} {save_msg}")

    print(f"\n Eğitim Tamamlandı! En Yüksek Başarı: %{best_acc:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title(f'ResNet-18 Performansı (En İyi: %{best_acc:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("resnet_sonucu.png")
    plt.show()

    plot_confusion_matrix(model, test_loader, TARGET_BIRDS)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n Eğitim durduruldu.")