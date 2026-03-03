import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// --- COMPONENTS ---

function ConfidenceBar({ label, value, description }) {
  if (value == null) return null;
  const pctNum = value * 100;
  const pct = pctNum.toFixed(1);
  
  const color =
    value < 0.4
      ? "from-red-500 to-rose-600"
      : value < 0.7
      ? "from-amber-400 to-orange-500"
      : "from-emerald-400 to-green-500";

  return (
    <div className="space-y-1 group relative">
      <div className="flex justify-between text-xs font-semibold tracking-wide text-zinc-400 uppercase mb-1">
        <span className="cursor-help" title={description}>{label}</span>
        <span className={value < 0.5 ? "text-red-400" : "text-emerald-400"}>{pct}% REAL</span>
      </div>
      <div className="h-2 w-full rounded-full bg-zinc-950 overflow-hidden border border-zinc-800">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pctNum}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={`h-full bg-gradient-to-r ${color} relative`}
        >
          <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite]" />
        </motion.div>
      </div>
    </div>
  );
}

function Card({ children, delay = 0, className = "" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay, ease: "easeOut" }}
      className={`rounded-2xl bg-zinc-900/40 backdrop-blur-xl border border-zinc-800/50 shadow-2xl p-6 ${className}`}
    >
      {children}
    </motion.div>
  );
}

// --- MAIN APP ---

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResults(null);
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await fetch("https://saiphanikrishna-omniscan-engine.hf.space/analyze", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Server Error");
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const isDeepfake = results?.is_deepfake;

  const downloadPDF = () => {
    // This triggers the browser's native, ultra-crisp PDF generator
    window.print();
  };

  return (
    <div className="min-h-screen bg-transparent text-zinc-100 flex items-center justify-center px-4 py-12 relative overflow-hidden font-sans">
      
      {/* 1. BACKGROUND (Hidden on PDF) */}
      <div className="fixed inset-0 pointer-events-none z-0 print:hidden">
        <img 
          src="image_0.jpg" 
          alt="Background Assay Network" 
          className="w-full h-full object-cover opacity-30 blur-sm brightness-50" 
        />
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-900/10 blur-[120px] rounded-full mix-blend-screen z-0" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-indigo-900/10 blur-[120px] rounded-full mix-blend-screen z-0" />
      </div>

      <div className="max-w-5xl w-full space-y-8 relative z-10">
        
        {/* Header (Hidden on PDF) */}
        <motion.header
          initial={{ opacity: 0, y: -15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex flex-col items-center gap-4 text-center print:hidden"
        >
          <div className="inline-flex items-center gap-2 rounded-full bg-zinc-900/80 backdrop-blur px-4 py-1.5 border border-zinc-800 shadow-lg">
            <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
            <span className="text-xs font-bold tracking-[0.2em] text-zinc-400 uppercase">
              OmniScan Engine
            </span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
            Enterprise Multimodal <br className="hidden sm:block" />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400">
              Deepfake Detection
            </span>
          </h1>
        </motion.header>

        {/* Upload + Loading (Hidden on PDF) */}
        <div className="print:hidden">
          <Card className="max-w-2xl mx-auto">
            <form onSubmit={handleUpload} className="flex flex-col gap-6">
              
              <div 
                onClick={() => !loading && fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl px-6 py-10 flex flex-col items-center justify-center text-center gap-3 transition-all duration-300 ${
                  loading ? 'border-zinc-800 bg-zinc-900/20 cursor-not-allowed' 
                  : file ? 'border-blue-500/50 bg-blue-500/5 hover:bg-blue-500/10 cursor-pointer' 
                  : 'border-zinc-700 hover:border-blue-500/50 hover:bg-zinc-800/50 cursor-pointer'
                }`}
              >
                <h2 className="text-lg sm:text-xl font-semibold text-white">
                  Drop media here or <span className="text-blue-400">browse files</span>
                </h2>
                <p className="text-xs sm:text-sm text-zinc-500 max-w-sm">
                  OmniScan supports MP4, MP3, WAV, JPG, and PNG. The engine will extract features and run ViT + SE‑ResNet ensembles.
                </p>
                
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="video/*,audio/*,image/*"
                  className="hidden"
                  disabled={loading}
                  onChange={(e) => setFile(e.target.files[0] ?? null)}
                />
                
                {file && (
                  <div className="mt-4 px-4 py-2 bg-zinc-950 rounded-lg border border-zinc-800 flex items-center gap-3">
                    <span className="font-medium truncate max-w-[200px] text-zinc-300">{file.name}</span>
                    <span className="text-zinc-600">•</span>
                    <span className="text-zinc-500 text-sm">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                )}
              </div>

              <div className="flex justify-center">
                <button
                  type="submit"
                  disabled={!file || loading}
                  className={`w-full sm:w-auto inline-flex justify-center items-center gap-2 rounded-xl px-8 py-3.5 text-sm font-bold transition-all duration-300 ${
                    !file || loading
                      ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                      : "bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_20px_rgba(37,99,235,0.3)] hover:shadow-[0_0_30px_rgba(37,99,235,0.5)]"
                  }`}
                >
                  {loading ? (
                    <>
                      <span className="h-4 w-4 rounded-full border-2 border-zinc-400 border-t-transparent animate-spin" />
                      Analyzing Tensors...
                    </>
                  ) : (
                    "Run Security Scan"
                  )}
                </button>
              </div>
            </form>

            <AnimatePresence>
              {loading && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto", marginTop: 24 }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex flex-col items-center gap-3 overflow-hidden"
                >
                  <p className="text-sm font-medium text-blue-400 animate-pulse">
                    Sequencing multimodal signal... extracting facial artifacts...
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>
        </div>

        {/* Results (THIS WILL PRINT) */}
        <AnimatePresence>
          {results && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="space-y-6"
            >
              <div id="scan-results" className="space-y-6 bg-transparent p-2 print:p-0 print:m-0 w-full [-webkit-print-color-adjust:exact] [print-color-adjust:exact]">
                
                <div className="grid gap-6 md:grid-cols-2">
                  
                  {/* Verdict Card */}
                  <Card delay={0.1} className={isDeepfake ? "border-red-500/30 print:border-red-500/50" : "border-emerald-500/30 print:border-emerald-500/50"}>
                    <p className="text-[11px] font-bold tracking-[0.25em] text-zinc-500 uppercase mb-3">
                      System Verdict
                    </p>
                    
                    <div className="flex flex-wrap items-center gap-3 mb-4">
                      <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-bold border ${
                          isDeepfake ? "bg-red-500/10 text-red-400 border-red-500/20" : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        }`}
                      >
                        <span className={`h-2 w-2 rounded-full ${isDeepfake ? 'bg-red-500 animate-pulse' : 'bg-emerald-500'}`} />
                        {isDeepfake ? "THREAT DETECTED" : "GENUINE SIGNAL"}
                      </span>
                      <span className="text-[11px] text-zinc-500 bg-zinc-900 px-2 py-1 rounded-md border border-zinc-800">
                        {results.media_type}
                      </span>
                    </div>
                    
                    <p className="text-base text-zinc-300 mb-4 font-medium leading-relaxed">
                      {results.diagnosis}
                    </p>
                    
                    {Array.isArray(results.frames) && results.frames.length > 0 && (
                      <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 text-xs text-zinc-400">
                        Frame sequence assay: Found{" "}
                        <span className="font-bold text-red-400">
                          {results.frames.filter((f) => f.is_fake === true).length}
                        </span>{" "}
                        synthetic segments out of{" "}
                        <span className="font-bold text-zinc-200">{results.frames.length}</span> keyframes.
                      </div>
                    )}
                  </Card>

                  {/* Confidence Card */}
                  <Card delay={0.2}>
                    <p className="text-[11px] font-bold tracking-[0.25em] text-zinc-500 uppercase mb-6">
                      Sub‑System Confidence
                    </p>
                    <div className="space-y-6">
                      {results.video_confidence !== null && (
                        <ConfidenceBar
                          label="Vision (ViT‑B)"
                          value={results.video_confidence}
                          description="Analyzes spatial anomalies and facial artifacts across extracted frames."
                        />
                      )}
                      {results.audio_confidence !== null && (
                        <ConfidenceBar
                          label="Audio (SE‑ResNet)"
                          value={results.audio_confidence}
                          description="Analyzes mel-spectrogram frequencies for acoustic manipulation."
                        />
                      )}
                      {results.video_confidence === null && results.audio_confidence === null && (
                        <p className="text-sm text-zinc-500 italic">No confidence metrics available for this format.</p>
                      )}
                    </div>
                  </Card>
                </div>

                {/* Heatmap Frames Grid */}
                {Array.isArray(results.frames) && results.frames.length > 0 && (
                  <Card delay={0.3}>
                    <div className="flex items-center justify-between mb-6">
                      <p className="text-[11px] font-bold tracking-[0.25em] text-zinc-500 uppercase">
                        XAI Artifact Extraction
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                      {results.frames.map((frame, idx) => (
                        <motion.div
                          key={idx}
                          whileHover={{ y: -5, scale: 1.05 }}
                          className={`relative rounded-xl overflow-hidden border-2 cursor-crosshair ${
                            frame.is_fake
                              ? "border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.2)] print:border-red-500"
                              : "border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.1)] print:border-emerald-500"
                          }`}
                        >
                          <img
                            src={frame.image_base64}
                            alt={`Frame ${idx}`}
                            className="w-full aspect-square object-cover"
                          />
                          <div className="absolute bottom-0 inset-x-0 px-2 py-1.5 flex justify-between items-center bg-black/80 backdrop-blur-sm border-t border-zinc-800">
                            <span className={`text-[10px] font-black tracking-wider ${frame.is_fake ? "text-red-400" : "text-emerald-400"}`}>
                              {frame.is_fake ? "FAKE" : "REAL"}
                            </span>
                            <span className="text-[10px] text-zinc-300 font-bold">
                              {(frame.prob_real * 100).toFixed(0)}%
                            </span>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </Card>
                )}
              </div>

              {/* EXPORT BUTTON (Hidden on the final PDF!) */}
              <motion.div 
                initial={{ opacity: 0 }} 
                animate={{ opacity: 1 }} 
                transition={{ delay: 0.5 }}
                className="flex justify-center mt-8 print:hidden"
              >
                <button
                  onClick={downloadPDF}
                  className="inline-flex items-center gap-3 px-6 py-3 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-white text-sm font-semibold rounded-xl shadow-lg transition-all duration-200 cursor-pointer"
                >
                  <span className="text-lg">📄</span>
                  Export Security Report (PDF)
                </button>
              </motion.div>

            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;