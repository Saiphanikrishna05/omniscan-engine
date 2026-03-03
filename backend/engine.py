import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import librosa
from PIL import Image
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import warnings
import base64

# --- HEATMAP IMPORTS ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

warnings.filterwarnings('ignore')

# --- 1. ARCHITECTURES ---
class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, max(1, out_channels // 8)),
            nn.ReLU(),
            nn.Linear(max(1, out_channels // 8), out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        y = self.se(out).unsqueeze(-1).unsqueeze(-1)
        return F.relu(out * y + residual)

class AudioDeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = SEResidualBlock(64, 64)
        self.drop1 = nn.Dropout2d(0.2) 
        self.res2 = SEResidualBlock(64, 128)
        self.drop2 = nn.Dropout2d(0.3) 
        self.res3 = SEResidualBlock(128, 256)
        self.drop3 = nn.Dropout2d(0.4) 
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.7), 
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.res1(x)
        x = self.drop1(x)
        x = self.res2(x)
        x = self.drop2(x)
        x = self.res3(x)
        x = self.drop3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class VideoDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoDeepfakeModel, self).__init__()
        self.model = models.vit_b_16(weights=None)
        self.model.heads = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        return self.model(x)


# --- 2. THE PIPELINE ENGINE ---
class DeepfakeFusionEngine:
    def __init__(self, video_weights_path: str, audio_weights_path: str, device: torch.device):
        self.device = device
        self.threshold = 0.50
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apple Silicon Fix for MTCNN
        mtcnn_device = torch.device("cpu") if self.device.type == "mps" else self.device
        self.mtcnn = MTCNN(margin=40, keep_all=False, post_process=False, device=mtcnn_device)
        
        # Load Video Model
        self.video_model = VideoDeepfakeModel(num_classes=2).to(self.device)
        self.video_model.load_state_dict(torch.load(video_weights_path, map_location=self.device, weights_only=True))
        self.video_model.eval()

        # Load Audio Model
        self.audio_model = AudioDeepfakeModel().to(self.device)
        self.audio_model.load_state_dict(torch.load(audio_weights_path, map_location=self.device, weights_only=True))
        self.audio_model.eval()
        
        # --- EXPLAINABLE AI (GRAD-CAM) SETUP ---
        self.cam = None
        if HAS_GRAD_CAM:
            try:
                # Target the final normalization layer in the Vision Transformer
                target_layers = [self.video_model.model.encoder.layers[-1].ln_1]
                
                # ViTs require reshaping the tokens back into a 2D grid for the heatmap
                def reshape_transform(tensor, height=14, width=14):
                    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                    result = result.transpose(2, 3).transpose(1, 2)
                    return result
                    
                self.cam = GradCAM(model=self.video_model, target_layers=target_layers, reshape_transform=reshape_transform)
                print(f"🛡️ Engine Online on {self.device} (Heatmaps: ENABLED)")
            except Exception as e:
                print(f"🛡️ Engine Online on {self.device} (Heatmaps: DISABLED - {e})")
        else:
            print(f"🛡️ Engine Online on {self.device} (Heatmaps: DISABLED - grad-cam not installed)")

    # --- 3. CORE PROCESSING HELPER ---
    # --- 3. CORE PROCESSING HELPER ---
    def process_face(self, face_np):
        """Runs the face through ViT and generates a heatmap if possible."""
        
        # THE FIX: Resize the MTCNN face (160x160) to match the AI Model and Heatmap (224x224)
        face_resized = cv2.resize(face_np, (224, 224))
        
        face_transformed = self.val_transform(Image.fromarray(face_resized)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.video_model(face_transformed)
            prob_real = torch.softmax(logits, dim=1)[:, 1].item()
            
        final_img = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
        
        # Try to overlay the Grad-CAM Heatmap
        if self.cam is not None:
            try:
                # Heatmap generates at 224x224
                grayscale_cam = self.cam(input_tensor=face_transformed, targets=None)[0, :]
                
                # Base image is now exactly 224x224!
                rgb_img = np.float32(face_resized) / 255.0 
                
                heatmap_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                final_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR) 
            except Exception as e:
                print(f"Heatmap Failed: {e}") 
                
        _, buffer = cv2.imencode('.jpg', final_img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        
        return prob_real, f"data:image/jpeg;base64,{b64_str}"

    # --- 4. ROUTING ENDPOINTS ---
    def analyze_audio(self, audio_path: str):
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
            if len(y) == 0: return None
        except Exception:
            return None
            
        target_samples = 16000 * 3
        y = np.pad(y, (0, target_samples - len(y))) if len(y) < target_samples else y[:target_samples]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=375, n_fft=1024)
        mel_db = librosa.power_to_db(mel, ref=np.max)[:, :128]
        
        tensor = torch.tensor(np.stack([mel_db, librosa.feature.delta(mel_db), librosa.feature.delta(mel_db, order=2)])).float()
        for c in range(tensor.size(0)):
            mean, std = tensor[c].mean(), tensor[c].std()
            tensor[c] = (tensor[c] - mean) / std if std > 1e-6 else tensor[c] - mean
            
        with torch.no_grad():
            logits = self.audio_model(tensor.unsqueeze(0).to(self.device))
            prob_real = torch.sigmoid(logits).item()
            
        return prob_real

    def scan_image_only(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None: return {"diagnosis": "Error reading image", "is_deepfake": False, "video_confidence": None, "audio_confidence": None}
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_tensor = self.mtcnn(Image.fromarray(img_rgb))
        
        if face_tensor is None: return {"diagnosis": "No human faces detected", "is_deepfake": False, "video_confidence": None, "audio_confidence": None}
             
        face_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
        prob_real, b64_img = self.process_face(face_np)
        
        is_fake = prob_real < self.threshold
        
        return {
            "video_confidence": prob_real,
            "audio_confidence": None,
            "is_deepfake": is_fake,
            "diagnosis": "Visual Manipulation Detected" if is_fake else "Genuine Media",
            "frames": [{"frame_num": 0, "prob_real": prob_real, "is_fake": is_fake, "image_base64": b64_img}]
        }

    def scan_audio_only(self, audio_path: str):
        prob_real = self.analyze_audio(audio_path)
        if prob_real is None: return {"diagnosis": "Error processing audio", "is_deepfake": False, "video_confidence": None, "audio_confidence": None}
             
        is_fake = prob_real < self.threshold
        return {
            "video_confidence": None,
            "audio_confidence": prob_real,
            "is_deepfake": is_fake,
            "diagnosis": "Acoustic Manipulation Detected" if is_fake else "Genuine Media",
            "frames": []
        }

    def scan_media(self, video_path: str):
        # 1. Process Video Frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // 8) if total_frames > 0 else 1
        
        face_probs, frame_data = [], []
        
        if total_frames > 0:
            for i in range(8):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                success, frame = cap.read()
                if not success: continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_tensor = self.mtcnn(Image.fromarray(frame_rgb))
                
                if face_tensor is not None:
                    face_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
                    prob_real, b64_img = self.process_face(face_np)
                    
                    face_probs.append(prob_real)
                    frame_data.append({
                        "frame_num": i * interval,
                        "prob_real": prob_real,
                        "is_fake": prob_real < self.threshold,
                        "image_base64": b64_img
                    })
            cap.release()
            
        vid_prob = sum(face_probs) / len(face_probs) if face_probs else 0.50
        
        # 2. Process Audio Track
        aud_prob = self.analyze_audio(video_path)
        aud_prob = aud_prob if aud_prob is not None else 0.50
        
        # 3. Final Diagnosis
        is_fake = (vid_prob < self.threshold) or (aud_prob < self.threshold)
        diagnosis = "Genuine Media"
        if is_fake:
            if vid_prob < self.threshold and aud_prob < self.threshold: diagnosis = "Full Synthesis (AI Face + AI Voice)"
            elif vid_prob < self.threshold: diagnosis = "Visual Manipulation Detected"
            else: diagnosis = "Acoustic Manipulation Detected"
                
        return {
            "video_confidence": vid_prob,
            "audio_confidence": aud_prob,
            "is_deepfake": is_fake,
            "diagnosis": diagnosis,
            "frames": frame_data 
        }