# OmniScan Engine 🛡️
**Enterprise Multimodal Deepfake Detection System**

OmniScan Engine is a full-stack security audit platform designed to detect synthetic media and deepfakes. It utilizes a multimodal machine learning architecture to analyze both visual anomalies and acoustic manipulation in uploaded media, providing explainable AI (XAI) outputs for end-users.

## 🚀 Key Features
* **Multimodal Analysis:** Seamlessly processes MP4, MP3, WAV, JPG, and PNG files.
* **Advanced Ensembles:** Leverages Vision Transformers (ViT) for spatial artifact extraction and SE-ResNet for audio mel-spectrogram analysis.
* **Explainable AI (XAI):** Generates real-time confidence heatmaps to highlight synthetic facial segments frame-by-frame.
* **Automated Security Audits:** Features client-side generation of professional PDF reports bypassing modern CSS constraints via native print routing.

## 🏗️ System Architecture
This project uses a modern monorepo structure separating the UI from the inference engine:
* **Frontend (`/frontend`):** React + Vite, styled with Tailwind CSS v4 and Framer Motion. Deployed on Vercel.
* **Backend (`/backend`):** Python-based API inference engine processing heavy `.pth` weights. Hosted seamlessly on Hugging Face Spaces.

## 👨‍💻 Author
**Saiphani Krishna Arumalla**
* GitHub: [@Saiphanikrishna05](https://github.com/Saiphanikrishna05)
