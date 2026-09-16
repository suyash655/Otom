"use client";
import { useState } from "react";

export default function BatchProcessingPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [processing, setProcessing] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFiles(Array.from(e.target.files));
  };

  const processBatch = async () => {
    setProcessing(true);
    const newResults = [];
    for (let i = 0; i < files.length; i++) {
      await new Promise(r => setTimeout(r, 150));
      const file = files[i];
      const conf = (Math.random() * (0.99 - 0.65) + 0.65).toFixed(3);
      newResults.push({ filename: file.name, prediction: "Normal", confidence: conf, needsReview: parseFloat(conf) < 0.75 ? "Yes" : "No" });
    }
    setResults(newResults);
    setProcessing(false);
  };

  const downloadCSV = () => {
    if (results.length === 0) return;
    const rows = results.map(r => `${r.filename},${r.prediction},${r.confidence},${r.needsReview}`).join("\n");
    const blob = new Blob(["Filename,Prediction,Confidence,Needs Review\n" + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `batch_results.csv`; a.click(); URL.revokeObjectURL(url);
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "var(--space-8)" }}>
      <h1>Batch Inference</h1>
      <input type="file" multiple accept="image/*, .dcm" onChange={handleFileChange} />
      <button onClick={processBatch} disabled={files.length === 0 || processing}>Process</button>
      {results.length > 0 && <button onClick={downloadCSV}>Download CSV</button>}
    </div>
  );
}
