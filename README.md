# 🖥️ Monitor Detection AI

Deteksi status monitor **nyala / mati** secara real-time menggunakan AI (YOLO11n) yang berjalan langsung di browser — **100% privat, tanpa server**.

## ✨ Fitur

- 🎯 Deteksi: `menyala`, `mati`, `objects`
- 📷 Kamera real-time + upload gambar
- 🖱️ Drag & drop gambar
- 📸 Snapshot dengan bounding box
- ⚡ Inferensi di browser (ONNX Runtime Web)
- 🔒 Tanpa upload data ke server

## 🚀 Deploy

### GitHub + Vercel (Recommended)

1. **Push ke GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/USERNAME/monitor-detection.git
   git push -u origin main
   ```

2. **Deploy ke Vercel:**
   - Buka [vercel.com](https://vercel.com)
   - Import repository GitHub ini
   - Framework: **Vite** (atau Other)
   - Build command: `npm run build`
   - Output directory: `dist`
   - Klik **Deploy** ✅

   > Header CORS sudah dikonfigurasi di `vercel.json` untuk ONNX Runtime.

### Development Lokal

```bash
npm install
npm run dev
```

Buka http://localhost:5173

## 📁 Struktur

```
monitor-detection/
├── index.html              # Main HTML
├── src/
│   ├── main.js             # Logika deteksi (ONNX inference)
│   └── style.css           # UI styling
├── public/
│   └── model/
│       └── best.onnx       # Model YOLO11n
├── vercel.json             # CORS headers untuk Vercel
├── vite.config.js          # Vite config
└── package.json
```

## 🧠 Model Info

| Property | Value |
|----------|-------|
| Architecture | YOLO11n |
| Format | ONNX (IR v9, Opset 20) |
| Input | `[1, 3, 640, 640]` |
| Output | `[1, 7, 8400]` |
| Classes | `mati`, `menyala`, `objects` |
| Conf threshold | 0.25 |
| NMS IoU | 0.45 |

## ⚠️ Catatan

- File `public/model/best.onnx` (~11MB) sudah termasuk di repo
- ONNX Runtime menggunakan **WebGL** jika tersedia, fallback ke **WASM**
- Header `COOP` + `COEP` wajib untuk SharedArrayBuffer (WASM threads)
- `vercel.json` sudah mengatur header ini secara otomatis

## 📄 Lisensi

Model: AGPL-3.0 (Ultralytics)
